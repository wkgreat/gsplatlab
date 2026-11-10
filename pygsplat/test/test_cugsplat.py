import argparse
import os

from pygsplat import SplatNumpyData, SplatRenderCuda, SplatScene

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Ceramic_Studio.ply", help="ply 数据路径，相对仓库根目录")
parser.add_argument("--scene", default="data/Ceramic_Studio.scene.json", help="scene.json 路径，相对仓库根目录")
parser.add_argument("--output", default="pygsplat/test/output/test_cugsplat.png", help="输出图片路径，相对仓库根目录")
args = parser.parse_args()

dataPath = os.path.join(repo_dir, args.data)
scenePath = os.path.join(repo_dir, args.scene)
imagePath = os.path.join(repo_dir, args.output)
os.makedirs(os.path.dirname(imagePath), exist_ok=True)

scene = SplatScene.from_file(scenePath)
render = SplatRenderCuda()
data = SplatNumpyData.from_ply(dataPath)
render.set_splat_data(data)
render.scene = scene
# viewer = SplatViewer(render, scene)

render.render_to_image(imagePath, flipY=True)
