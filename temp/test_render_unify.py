import numpy as np
from pygsplat import (
    SplatNumpyData,
    SplatRender,
    SplatRenderCuda,
    SplatRenderOpenGL,
    SplatScene,
    SplatViewer,
    splats_from_ply,
)

scene_gl = SplatScene.from_file("../../data/cactus_splat3.scene.json")
render_gl = SplatRenderOpenGL(scene_gl.width, scene_gl.height, scene_gl)
render_gl.set_splat_data(splats_from_ply("../../data/cactus_splat3.ply"))
assert isinstance(render_gl, SplatRender)
assert type(render_gl).render is not SplatRender.render
image_gl = render_gl.render()
print("GL", image_gl.shape, image_gl.dtype, float(image_gl.min()), float(image_gl.max()))
render_gl.render_to_image("output/glviewer.png", flipY=True)

scene_cu = SplatScene.from_file("../../data/cactus_splat3.scene.json")
render_cu = SplatRenderCuda(scene_cu.width, scene_cu.height, scene_cu)
render_cu.set_splat_data(SplatNumpyData.from_ply("../../data/cactus_splat3.ply"))
assert isinstance(render_cu, SplatRender)
assert type(render_cu).render is not SplatRender.render
image_cu = render_cu.render()
print("CU", image_cu.shape, image_cu.dtype, float(image_cu.min()), float(image_cu.max()))
render_cu.render_to_image("output/cuviewer.png", flipY=True)

viewer = SplatViewer(render_gl, scene_gl)
print("viewer", viewer.canvas.width, viewer.canvas.height, type(render_gl).__name__)