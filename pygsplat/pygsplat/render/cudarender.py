import numpy as np

from ..functions import gsplat_rasterize
from .render import SplatRender


class SplatRenderCuda(SplatRender):
    """CUDA 光栅化后端。"""

    def render(self) -> np.ndarray:
        return gsplat_rasterize(
            self.scene,
            self.data.pos3d,
            self.data.scale,
            self.data.rot,
            self.data.opacity,
            self.data.sh,
        )
