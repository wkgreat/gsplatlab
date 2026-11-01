import os
from pathlib import Path

# editable 安装时 _C 扩展和 cugsplat 运行时库由 setup.py 统一产出到
# <repo>/build/pygsplat，这里把它接进包的搜索路径，避免污染源码树。
_native_dir = Path(__file__).resolve().parents[2] / "build" / "pygsplat"
_dll_dir_handle = None
if any(_native_dir.glob("_C*")):
    if str(_native_dir) not in __path__:
        __path__.append(str(_native_dir))
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _dll_dir_handle = os.add_dll_directory(str(_native_dir))


from . import render
from .dataloader import SplatNumpyData
from .functions import add_one, gsplat_compute_debug, gsplat_rasterize
from .image import save_to_image
from .render import SplatRender, SplatRenderCuda, SplatRenderOpenGL
from .scene import SplatScene, SplatSceneWidget
from .viewer import SplatViewer

__all__ = [
    "SplatNumpyData",
    "SplatRender",
    "SplatRenderCuda",
    "SplatRenderOpenGL",
    "SplatScene",
    "SplatSceneWidget",
    "SplatViewer",
    "add_one",
    "gsplat_compute_debug",
    "gsplat_rasterize",
    "save_to_image",
]
