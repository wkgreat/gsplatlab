import os
from pygsplat import SplatNumpyData, SplatRenderCuda, SplatScene, SplatViewer

script_dir = os.path.dirname(os.path.abspath(__file__))

dataName = "cactus_splat3.ply"
sceneName = "cactus_splat3.scene.json"

dataPath = os.path.join(script_dir, f"../../data/{dataName}")
scenePath = os.path.join(script_dir, f"../../data/{sceneName}")
scene = SplatScene.from_file(scenePath)
render = SplatRenderCuda()
data = SplatNumpyData.from_ply(dataPath)
render.set_splat_data(data)
render.scene = scene
# viewer = SplatViewer(render, scene)

imagePath = os.path.join(script_dir, "output/test_cugsplat.png")
render.render_to_image(imagePath, flipY=True)
