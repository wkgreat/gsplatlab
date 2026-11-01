import json
from pathlib import Path

nb_path = Path(r"pygsplat/test/cuviewer.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]

# 1) import SplatSceneWidget
cell0 = "".join(cells[0]["source"])
old_import = "from pygsplat import SplatScene, gsplat_rasterize, save_to_image"
new_import = (
    "from pygsplat import (\n"
    "    SplatScene,\n"
    "    SplatSceneWidget,\n"
    "    gsplat_rasterize,\n"
    "    save_to_image,\n"
    ")"
)
if old_import not in cell0:
    raise SystemExit("import not found")
cells[0]["source"] = cell0.replace(old_import, new_import).splitlines(keepends=True)

# 2) replace SplatRender / SplatViewer with scene-based implementations
src = "".join(cells[1]["source"])
idx = src.index("class SplatRender:")
new_classes = '''class SplatRender:
    def __init__(self, width, height) -> None:
        self.scene = SplatScene()
        self.scene.width = width
        self.scene.height = height
        # 3DGS 数据通常是 Z-up
        self.scene.reference_up = glm.vec3(0, 0, 1)
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
        return image

    def render_to_image(self, image_path, flipY=False):
        arr = self.render()
        save_to_image(arr, image_path, 255, flipY)


class SplatViewer:
    def __init__(self, width, height):
        self.render = SplatRender(width, height)
        self.canvas = Canvas(width=width, height=height)

    def set_splat_data(self, data):
        self.render.setSplatData(data)

    def render_to_canvas(self):
        image = self.render.render()
        image = (image * 255).astype(np.uint8)
        with hold_canvas(self.canvas):
            self.canvas.put_image_data(np.flipud(image))

    def show(self):
        scene_widget = SplatSceneWidget(self.render.scene)
        scene_widget.on_change(self.render_to_canvas)
        scene_widget.bind_canvas(self.canvas)
        display(widgets.HBox([self.canvas, scene_widget.panel]))
        self.render_to_canvas()
'''
cells[1]["source"] = (src[:idx] + new_classes).splitlines(keepends=True)

# 3) main cell: replace old orbit API calls
main = "".join(cells[2]["source"])
old_setup = '''viewer = SplatViewer(width, height)
viewer.render.set_orbit_x(orbit[0])
viewer.render.set_orbit_y(orbit[1])
viewer.render.set_orbit_r(radius)
viewer.render.set_center(center)
'''
new_setup = '''viewer = SplatViewer(width, height)
scene = viewer.render.scene

# orbit/radius 转成相机位置，center 由 SplatScene 管理
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
'''
if old_setup not in main:
    raise SystemExit("old setup not found")
cells[2]["source"] = main.replace(old_setup, new_setup).splitlines(keepends=True)

# 4) drop the trailing crashed cell and clear stale outputs
nb["cells"] = cells[:3]
for cell in nb["cells"]:
    cell["outputs"] = []
    cell["execution_count"] = None

nb_path.write_text(
    json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print("updated")
