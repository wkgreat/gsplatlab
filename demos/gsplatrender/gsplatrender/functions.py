import numpy as np

from . import _C


def add_one(a: np.ndarray) -> np.ndarray:
    return _C.add_one(a)


def gsplat_rasterize(
    scene,
    pos3d: np.ndarray,
    scale: np.ndarray,
    rot: np.ndarray,
    opacity: np.ndarray,
    sh: np.ndarray,
) -> np.ndarray:
    return _C.gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh)


def gsplat_compute_debug(
    scene,
    pos3d: np.ndarray,
    scale: np.ndarray,
    rot: np.ndarray,
    opacity: np.ndarray,
    sh: np.ndarray,
):
    """返回 compute 阶段的中间结果: (pos2d, depth, cov2d, rect, color)"""
    return _C.gsplat_compute_debug(scene, pos3d, scale, rot, opacity, sh)
