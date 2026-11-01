import json
import math
import sys

sys.path.insert(0, r"pygsplat")
from pyglm import glm

nb = json.load(open(r"pygsplat/test/cuviewer.ipynb", encoding="utf-8"))
ns = {}
for cell in nb["cells"][:2]:
    exec("".join(cell["source"]), ns)

SplatViewer = ns["SplatViewer"]
viewer = SplatViewer(500, 500)
scene = viewer.render.scene

orbit = [-90.0, -42.0]
radius = 2.0
center = [0, 0, 2]
yaw = math.radians(orbit[0])
pitch = math.radians(orbit[1])
scene.center = glm.vec3(*center)
scene.set_camera_pos(
    scene.center
    + glm.vec3(
        radius * math.cos(pitch) * math.cos(yaw),
        radius * math.cos(pitch) * math.sin(yaw),
        radius * math.sin(pitch),
    )
)

print("eye", tuple(round(v, 4) for v in scene.eye))
print("center", tuple(scene.center))
print("up", tuple(round(v, 4) for v in scene.up))
print("angles", tuple(round(v, 4) for v in scene.camera_angles()))
print("dist", round(glm.length(scene.eye - scene.center), 4))
print("viewmtx", scene.viewmtx.tolist())

w = ns["SplatSceneWidget"](scene)
print("info ok", "up     :" in w.info.value)
