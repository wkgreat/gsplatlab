import numpy as np
from plyfile import PlyData


class SplatNumpyData:
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
        data = SplatNumpyData(pos3d, scale, rot, opacity, sh)

        return data


class SplatTensorData:
    pass
