from importlib import resources
from typing import Any, cast

import moderngl
import numpy as np
from pyglm import glm

from ..dataloader import SplatNumpyData
from .render import SplatRender


def pack_splat_data(data: SplatNumpyData) -> np.ndarray:
    """把 SplatNumpyData 打包成 OpenGL SSBO 的 (N, 76) 布局。"""
    nsplats = data.pos3d.shape[0]

    # 3 position + 1 opacity -> GLSL vec4 posopa
    posopa = np.concatenate([data.pos3d[:, :3], data.opacity], axis=1)

    # GLSL 是 vec4 shcolor[16]：RGB 后补一个 float，保证每个系数占 16 bytes
    sh_vec4 = np.zeros((nsplats, 16, 4), dtype=np.float32)
    sh_vec4[:, :, :3] = data.sh.reshape(nsplats, 16, 3)

    # 每条记录：posopa + scale + rot + 16 个 vec4 = 76 float32 = 304 bytes
    splats = np.concatenate(
        [posopa, data.scale, data.rot, sh_vec4.reshape(nsplats, 64)],
        axis=1,
    )
    assert splats.shape == (nsplats, 76)
    return np.ascontiguousarray(splats, dtype=np.float32)


def get_uniform(shader: Any, name: str) -> moderngl.Uniform:
    return cast(moderngl.Uniform, shader[name])


class SplatRenderOpenGL(SplatRender):
    """OpenGL compute + instanced rasterization 后端。"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        scene=None,
    ) -> None:
        super().__init__(width, height, scene)
        if scene is None:
            # 默认 Z-up，eye=(0, 0, 40)，与旧版默认视角一致
            self.scene.eye = glm.vec3(0, 0, 40)
            self.scene.reference_up = glm.vec3(0, 0, 1)
            self.scene.up = glm.vec3(0, 0, 1)
            self.scene.update()

        self.create_context()
        self.nsplats = 0
        self.vao = None
        self.input_ssbo = None
        self.ssbo = None

        self.vertcode = self.read_shader("splat.vert.glsl")
        self.fragcode = self.read_shader("splat.frag.glsl")
        self.program = self.ctx.program(
            vertex_shader=self.vertcode,
            fragment_shader=self.fragcode,
        )

        self.compcode = self.read_shader("splat.comp.glsl")
        self.compute_program = self.ctx.compute_shader(self.compcode)
        self.sortcode = self.read_shader("splat_sort.glsl")
        self.sort_program = self.ctx.compute_shader(self.sortcode)

        self.fbo = None
        self._resize_framebuffer()

        self.ctx.point_size = 1.0

    def read_shader(self, path):
        # 从安装后的包目录读取 shader：editable 与 wheel 安装都能解析到
        return resources.files(__package__).joinpath(path).read_text(encoding="utf-8")

    def create_context(self):
        # 服务器 / Docker / 无显示器环境优先尝试 EGL
        try:
            self.ctx = moderngl.create_context(standalone=True, backend="egl")
        except Exception:
            # 本地桌面环境可尝试默认 backend
            self.ctx = moderngl.create_context(standalone=True)

    def _resize_framebuffer(self):
        size = (int(self.scene.width), int(self.scene.height))
        if self.fbo is not None and tuple(self.fbo.size) == size:
            return
        self.fbo = self.ctx.simple_framebuffer(size, components=4)
        self.fbo.use()
        self.pixel_buffer = bytearray(size[0] * size[1] * 4)

    def set_splat_data(self, data):
        # 统一入口：既接受 SplatNumpyData，也接受已打包好的 (N, 76) 数组
        if isinstance(data, SplatNumpyData):
            splats = pack_splat_data(data)
        else:
            splats = np.ascontiguousarray(data, dtype=np.float32)

        super().set_splat_data(splats)
        self.nsplats = splats.shape[0]
        self.sort_count = (
            1 if self.nsplats == 0 else 1 << (self.nsplats - 1).bit_length()
        )
        self.input_ssbo = self.ctx.buffer(splats.tobytes())
        self.input_ssbo.bind_to_storage_buffer(binding=0)
        output_splats = np.zeros((self.sort_count, 16), dtype=np.float32)
        output_splats[self.nsplats :, 2] = -np.finfo(np.float32).max
        self.ssbo = self.ctx.buffer(output_splats.tobytes())
        self.ssbo.bind_to_storage_buffer(binding=1)
        self.vao = self.ctx.vertex_array(self.program, [])

    def sortSplats(self):
        # 普通 alpha blending 要求 splat 按相机深度从远到近绘制。
        positions = self.data[:, :3]
        eye = np.array(self.scene.eye, dtype=np.float32)
        center = np.array(self.scene.center, dtype=np.float32)
        forward = center - eye
        forward_length = np.linalg.norm(forward)
        if forward_length <= np.finfo(np.float32).eps:
            raise ValueError("scene.eye 和 scene.center 不能是同一个位置")
        forward /= forward_length

        # 相机前方的 depth 为正；数值越大表示离相机越远。
        depth = (positions - eye) @ forward
        self.sorted_indices = np.argsort(-depth, kind="stable")
        sorted_splats = np.ascontiguousarray(
            self.data[self.sorted_indices], dtype=np.float32
        )
        return sorted_splats

    def sortSplatsGPU(self):
        if self.nsplats <= 1:
            return

        shader = self.sort_program
        shader["sort_count"].value = int(self.sort_count)

        self.ssbo.bind_to_storage_buffer(binding=0)
        group_count = (self.sort_count + 255) // 256
        k = 2
        while k <= self.sort_count:
            j = k // 2
            while j > 0:
                shader["k"].value = int(k)
                shader["j"].value = int(j)
                shader.run(group_x=group_count)
                self.ctx.memory_barrier()
                j //= 2
            k *= 2

    def set_point_size(self, size):
        self.ctx.point_size = size

    def compute(self):
        if self.nsplats == 0:
            return

        cs = self.compute_program
        eye = self.scene.eye
        get_uniform(cs, "camera.eye").value = (eye.x, eye.y, eye.z, 1.0)
        # scene 里是 np.array(glm.mat4)（列主序视图），按 F 序取字节才是 GLSL 的列主序。
        get_uniform(cs, "camera.viewmtx").write(self.scene.viewmtx.tobytes(order="F"))
        get_uniform(cs, "projection.fovy").value = self.scene.fovy
        get_uniform(cs, "projection.projmtx").write(
            self.scene.projmtx.tobytes(order="F")
        )
        get_uniform(cs, "viewport.size").value = (
            float(self.scene.width),
            float(self.scene.height),
        )
        get_uniform(cs, "nsplats").value = int(self.nsplats)
        self.input_ssbo.bind_to_storage_buffer(binding=0)
        self.ssbo.bind_to_storage_buffer(binding=1)

        group_count = (self.nsplats + 255) // 256
        cs.run(group_x=group_count)

        self.ctx.memory_barrier()
        self.sortSplatsGPU()
        self.ctx.memory_barrier()

    def render(self) -> np.ndarray:
        self._resize_framebuffer()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.ctx.blend_func = (
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        self.ctx.clear(*self.scene.bgcolor.tolist(), depth=1.0)

        if self.nsplats > 0:
            self.compute()
            self.ssbo.bind_to_storage_buffer(binding=0)
            get_uniform(self.program, "viewport.size").value = (
                float(self.scene.width),
                float(self.scene.height),
            )
            self.vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=self.nsplats)

        self.fbo.read_into(self.pixel_buffer, components=4, alignment=1)
        img = np.frombuffer(self.pixel_buffer, dtype=np.uint8).reshape(
            self.scene.height, self.scene.width, 4
        )

        # 与 SplatRenderCuda 保持一致：输出 [0, 1] 的 float32 RGBA
        return img.astype(np.float32) / 255.0
