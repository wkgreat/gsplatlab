import numpy as np
from gsplatrender import gsplat_rasterize
from PIL import Image
from plyfile import PlyData
from pyglm import glm


class SplatScene:
    bgcolor = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    viewmtx = np.array(glm.mat4(1.0), dtype=np.float32)
    projmtx = np.array(glm.mat4(1.0), dtype=np.float32)
    eye = glm.vec3(1, 1, 1)
    center = glm.vec3(0, 0, 0)
    up = glm.vec3(0, 1, 0)
    near = 0.01
    far = 1000
    fovy = glm.pi() / 2.0
    width = 1024
    height = 768

    def update(self):
        _viewmtx = glm.lookAt(self.eye, self.center, self.up)
        self.viewmtx = np.array(_viewmtx, dtype=np.float32)
        _projmtx = glm.perspective(
            self.fovy, self.width * 1.0 / self.height, self.near, self.far
        )
        self.projmtx = np.array(_projmtx, dtype=np.float32)


class SplatsData:
    def __init__(self, pos3d, scale, rot, opacity, sh) -> None:
        self.pos3d = pos3d
        self.scale = scale
        self.rot = rot
        self.opacity = opacity
        self.sh = sh

    @staticmethod
    def from_ply(path):
        vertex = PlyData.read(path)["vertex"]
        nsplats = len(vertex["x"])

        # 3 position + 1 opacity -> GLSL vec4 posopa
        position = np.stack(
            [
                vertex["x"],
                vertex["y"],
                vertex["z"],
                np.ones((nsplats,), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(nsplats, 1)

        splat_scale = np.stack(
            [
                vertex["scale_0"],
                vertex["scale_1"],
                vertex["scale_2"],
                np.zeros((nsplats,), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        # PLY 中 rotation 为 (w, x, y, z)，与 shader 当前读取方式一致
        splat_rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(
            np.float32
        )

        # DC 是第 0 个 SH 系数，形状为 (N, 1, 3)
        sh_dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(
            np.float32
        )[:, None, :]

        # f_rest 按 RGB 通道分别保存：先恢复 (N, 3, 15)，再转为 (N, 15, 3)
        sh_rest = np.stack([vertex[f"f_rest_{i}"] for i in range(45)], axis=1).astype(
            np.float32
        )
        sh_rest = sh_rest.reshape(nsplats, 3, 15).transpose(0, 2, 1)

        # GLSL 是 vec4 shcolor[16]：RGB 后补一个 float，保证每个系数占 16 bytes
        sh_rgb = np.concatenate([sh_dc, sh_rest], axis=1)
        sh_vec4 = np.zeros((nsplats, 16, 4), dtype=np.float32)
        sh_vec4[:, :, :3] = sh_rgb
        sh_vec4 = sh_vec4.reshape(nsplats, 64)

        pos3d = np.ascontiguousarray(position, dtype=np.float32)
        scale = np.ascontiguousarray(splat_scale, dtype=np.float32)
        rot = np.ascontiguousarray(splat_rot, dtype=np.float32)
        opacity = np.ascontiguousarray(opacity, dtype=np.float32)
        sh = np.ascontiguousarray(sh_rgb.reshape(nsplats, 48), dtype=np.float32)
        data = SplatsData(pos3d, scale, rot, opacity, sh)

        return data


class SplatRenderer:
    def __init__(self, width, height) -> None:
        self.scene = SplatScene()
        self.scene.near = 0.01
        self.scene.far = 1000
        self.scene.fovy = glm.pi() / 2.0
        self.scene.width = width
        self.scene.height = height
        self.orbit_x = -90.0
        self.orbit_y = 0
        self.orbit_r = 2
        self.update_scene()

    def read_shader(self, path):
        source = ""
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return source

    def set_orbit_x(self, rad):
        self.orbit_x = rad
        self.update_scene()

    def set_orbit_y(self, rad):
        self.orbit_y = rad
        self.update_scene()

    def set_orbit_r(self, radius):
        self.orbit_r = radius
        self.update_scene()

    def update_scene(self):
        x = self.orbit_r * glm.cos(self.orbit_y) * glm.cos(self.orbit_x)
        y = self.orbit_r * glm.cos(self.orbit_y) * glm.sin(self.orbit_x)
        z = self.orbit_r * glm.sin(self.orbit_y)
        self.scene.eye = glm.vec3(x, y, z)
        self.scene.center = glm.vec3(0, 0, 0)
        self.scene.up = glm.vec3(0, 0, 1)
        self.scene.update()

    def setSplatData(self, splats):
        self.data = splats

    def render(self):

        image = gsplat_rasterize(
            self.scene,
            self.data.pos3d,
            self.data.scale,
            self.data.rot,
            self.data.opacity,
            self.data.sh,
        )
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save("output.png")


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    width = 500
    height = 500
    path = os.path.join(script_dir, "../../data/cactus_splat3.ply")
    data = SplatsData.from_ply(path)
    renderer = SplatRenderer(width, height)
    renderer.setSplatData(data)
    renderer.render()
