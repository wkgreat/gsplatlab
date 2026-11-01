import numpy as np

from ..dataloader import SplatNumpyData
from ..image import save_to_image
from ..scene import SplatScene


class SplatRender:
    """Splat 渲染器基类。

    子类（SplatRenderCuda / SplatRenderOpenGL）只需要准备自己的渲染资源，
    并重写 render()；场景、数据以及保存图片的流程在这里统一处理。
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        scene: SplatScene | None = None,
    ) -> None:
        self.scene = scene if scene is not None else SplatScene()
        self.scene.width = width
        self.scene.height = height
        self.data: SplatNumpyData | np.ndarray | None = None

    def set_scene(self, scene: SplatScene):
        self.scene = scene

    def set_splat_data(self, data: SplatNumpyData | np.ndarray):
        self.data = data

    def render(self) -> np.ndarray:
        """返回 HxWx4、float32、取值 [0, 1] 的 RGBA 图像。"""
        raise NotImplementedError

    def render_to_image(self, image_path, flipY=False):
        save_to_image(self.render(), image_path, 255, flipY)
