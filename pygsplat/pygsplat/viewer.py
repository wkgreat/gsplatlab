import ipywidgets as widgets
import numpy as np
from ipycanvas import Canvas, hold_canvas
from IPython.display import display

from .render.render import SplatRender
from .scene import SplatScene, SplatSceneWidget


class SplatViewer:
    def __init__(self, render: SplatRender, scene: SplatScene):
        self.render = render
        self.render.set_scene(scene)
        self.canvas = Canvas(width=scene.width, height=scene.height)

    def set_splat_data(self, data):
        self.render.set_splat_data(data)

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
