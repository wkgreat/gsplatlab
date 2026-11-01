import json
import math

import ipywidgets as widgets
import numpy as np
from pyglm import glm


class SplatScene:
    bgcolor = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    viewmtx = np.array(glm.mat4(1.0), dtype=np.float32)
    projmtx = np.array(glm.mat4(1.0), dtype=np.float32)
    eye = glm.vec3(1, 1, 1)
    center = glm.vec3(0, 0, 0)
    # roll=0 时参考的世界上方向；相机 up 绕视线旋转后产生 roll
    reference_up = glm.vec3(0, 1, 0)
    near = 0.01
    far = 1000
    fovy = glm.pi() / 2.0
    width = 1024
    height = 768

    def __init__(self):
        self.eye = glm.vec3(self.eye)
        self.center = glm.vec3(self.center)
        # 相机朝向的中间状态统一用四元数（相机空间 -> 世界空间）维护
        self.orientation = self._orientation_from_look(
            self.eye, self.center, glm.vec3(0, 1, 0)
        )
        self.update()

    @property
    def up(self):
        """相机 up，由 orientation 导出。"""
        return glm.normalize(self.orientation * glm.vec3(0, 1, 0))

    @up.setter
    def up(self, value):
        self.orientation = self._orientation_from_look(
            self.eye, self.center, glm.vec3(value)
        )

    def _orientation_from_look(self, eye, center, up):
        """由 lookAt 参数构造相机 -> 世界四元数。"""
        forward = center - eye
        if glm.length(forward) < 1e-9:
            forward = glm.vec3(0, 0, -1)
        else:
            forward = glm.normalize(forward)

        up = glm.normalize(glm.vec3(up))
        right = glm.cross(forward, up)
        if glm.length(right) < 1e-9:
            # 视线与 up 平行时退化，另选一个方向再求 right
            up = glm.cross(forward, glm.vec3(1, 0, 0))
            if glm.length(up) < 1e-9:
                up = glm.cross(forward, glm.vec3(0, 1, 0))
            up = glm.normalize(up)
            right = glm.cross(forward, up)
        right = glm.normalize(right)
        camera_up = glm.normalize(glm.cross(right, forward))

        # mat3 的三列分别是相机 right / up / back 在世界空间中的方向
        return glm.normalize(glm.quat_cast(glm.mat3(right, camera_up, -forward)))

    def _orientation_from_angles(self, head, pitch, roll):
        """由 head / pitch / roll（相对 reference_up）构造四元数。"""
        up = glm.normalize(self.reference_up)
        reference = glm.vec3(1, 0, 0) - up * glm.dot(glm.vec3(1, 0, 0), up)
        if glm.length(reference) < 1e-9:
            reference = glm.vec3(0, 1, 0) - up * glm.dot(glm.vec3(0, 1, 0), up)
        reference = glm.normalize(reference)
        side = glm.cross(up, reference)
        horizontal = reference * math.cos(head) + side * math.sin(head)
        forward = horizontal * math.cos(pitch) + up * math.sin(pitch)

        zero_up = up - forward * glm.dot(up, forward)
        if glm.length(zero_up) < 1e-9:
            # 视线与 reference_up 平行时退化，与 _orientation_from_look 保持一致
            zero_up = glm.cross(forward, glm.vec3(1, 0, 0))
            if glm.length(zero_up) < 1e-9:
                zero_up = glm.cross(forward, glm.vec3(0, 1, 0))
        zero_up = glm.normalize(zero_up)
        right = glm.normalize(glm.cross(forward, zero_up))
        camera_up = glm.normalize(glm.cross(right, forward))

        if roll:
            rotation = glm.angleAxis(roll, forward)
            right = glm.normalize(rotation * right)
            camera_up = glm.normalize(rotation * camera_up)

        return glm.normalize(glm.quat_cast(glm.mat3(right, camera_up, -forward)))

    def _camera_axes(self):
        """返回相机在世界空间中的 (forward, right, up)。"""
        return (
            -(self.orientation * glm.vec3(0, 0, 1)),
            self.orientation * glm.vec3(1, 0, 0),
            self.orientation * glm.vec3(0, 1, 0),
        )

    def update(self):
        # 世界 -> 相机：先平移到相机原点，再施加四元数的逆旋转
        view = glm.mat4_cast(glm.conjugate(self.orientation)) * glm.translate(
            glm.mat4(1.0), -self.eye
        )
        self.viewmtx = np.array(view, dtype=np.float32)
        _projmtx = glm.perspective(
            self.fovy, self.width * 1.0 / self.height, self.near, self.far
        )
        self.projmtx = np.array(_projmtx, dtype=np.float32)

    def set_viewport(self, width: int, height: int):
        """设置渲染分辨率，并立即重算依赖视口的状态（如投影矩阵）。"""
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("viewport width/height 必须大于 0")
        if self.width == width and self.height == height:
            return
        self.width = width
        self.height = height
        self.update()

    def camera_rotate(self, yaw: float = 0.0, pitch: float = 0.0):
        """绕 center 旋转相机：yaw 绕 reference_up，pitch 绕相机 right。"""
        distance = glm.length(self.center - self.eye)
        if distance < 1e-9:
            return

        if yaw:
            axis = glm.normalize(self.reference_up)
            self.orientation = glm.normalize(
                glm.angleAxis(yaw, axis) * self.orientation
            )
        if pitch:
            _, right, _ = self._camera_axes()
            self.orientation = glm.normalize(
                glm.angleAxis(pitch, right) * self.orientation
            )

        forward, _, _ = self._camera_axes()
        self.eye = self.center - forward * distance
        self.update()

    def camera_pitch_up(self, rad: float):
        """相机抬头。"""
        self.camera_rotate(pitch=rad)

    def camera_pitch_down(self, rad: float):
        """相机低头。"""
        self.camera_rotate(pitch=-rad)

    def camera_head_left(self, rad: float):
        """相机向左转头。"""
        self.camera_rotate(yaw=rad)

    def camera_head_right(self, rad: float):
        """相机向右转头。"""
        self.camera_rotate(yaw=-rad)

    def camera_pan(self, right: float = 0.0, up: float = 0.0):
        """在相机平面内平移：right 沿相机右方，up 沿相机上方。"""
        _, cam_right, cam_up = self._camera_axes()
        delta = cam_right * right + cam_up * up
        self.eye = self.eye + delta
        self.center = self.center + delta
        self.update()

    def camera_pan_left(self, amount: float):
        self.camera_pan(right=-amount)

    def camera_pan_right(self, amount: float):
        self.camera_pan(right=amount)

    def camera_pan_up(self, amount: float):
        self.camera_pan(up=amount)

    def camera_pan_down(self, amount: float):
        self.camera_pan(up=-amount)

    def camera_zoom(self, factor: float):
        """factor < 1 拉近，factor > 1 拉远。"""
        if factor <= 0:
            raise ValueError("zoom factor 必须大于 0")

        offset = self.eye - self.center
        if glm.length(offset) < 1e-6:
            return

        self.eye = self.center + offset * factor
        self.update()

    def camera_zoom_in(self, factor: float):
        self.camera_zoom(1.0 / factor)

    def camera_zoom_out(self, factor: float):
        self.camera_zoom(factor)

    def camera_angles(self):
        """返回 (distance, head, pitch, roll)，角度单位为弧度。"""
        forward = self.center - self.eye
        distance = glm.length(forward)
        if distance < 1e-9:
            return 0.0, 0.0, 0.0, 0.0

        forward = glm.normalize(forward)
        reference_up = glm.normalize(self.reference_up)

        pitch = math.asin(max(-1.0, min(1.0, glm.dot(forward, reference_up))))

        horizontal = forward - reference_up * glm.dot(forward, reference_up)
        if glm.length(horizontal) < 1e-9:
            head = 0.0
        else:
            horizontal = glm.normalize(horizontal)
            # 以世界 X 轴在水平面上的投影作为 head 的 0 度参考
            reference = glm.vec3(1, 0, 0) - reference_up * glm.dot(
                glm.vec3(1, 0, 0), reference_up
            )
            if glm.length(reference) < 1e-9:
                reference = glm.vec3(0, 1, 0) - reference_up * glm.dot(
                    glm.vec3(0, 1, 0), reference_up
                )
            reference = glm.normalize(reference)
            side = glm.cross(reference_up, reference)
            head = math.atan2(
                glm.dot(horizontal, side), glm.dot(horizontal, reference)
            )

        _, _, cam_up = self._camera_axes()
        world_up = reference_up - forward * glm.dot(reference_up, forward)
        if glm.length(world_up) < 1e-9:
            # 视线与 reference_up 平行时用固定方向作为 roll 参考
            world_up = glm.cross(forward, glm.vec3(1, 0, 0))
            if glm.length(world_up) < 1e-9:
                world_up = glm.cross(forward, glm.vec3(0, 1, 0))
        world_up = glm.normalize(world_up)
        roll = math.atan2(
            glm.dot(glm.cross(world_up, cam_up), forward),
            glm.dot(world_up, cam_up),
        )

        return distance, head, pitch, roll

    def _apply_angles(self, head, pitch, roll, distance):
        self.orientation = self._orientation_from_angles(head, pitch, roll)
        forward, _, _ = self._camera_axes()
        self.eye = self.center - forward * distance
        self.update()

    def set_head(self, rad: float):
        """设置水平朝向 head，保持 center、距离、pitch 和 roll 不变。"""
        distance, _, pitch, roll = self.camera_angles()
        if distance < 1e-9:
            return
        self._apply_angles(rad, pitch, roll, distance)

    def set_pitch(self, rad: float):
        """设置俯仰角 pitch（相对 reference_up），保持其余分量不变。"""
        half_pi = glm.pi() * 0.5
        distance, head, _, roll = self.camera_angles()
        if distance < 1e-9:
            return
        self._apply_angles(head, max(-half_pi, min(half_pi, rad)), roll, distance)

    def set_roll(self, rad: float):
        """设置绕视线方向的 roll，保持 eye/center 不变。"""
        distance, head, pitch, _ = self.camera_angles()
        if distance < 1e-9:
            return
        self._apply_angles(head, pitch, rad, distance)

    def set_camera_pos(self, x, y=None, z=None):
        """设置相机 eye 位置；支持 vec3/序列，或直接传 x, y, z。"""
        up = self.up
        if y is None and z is None:
            self.eye = glm.vec3(x)
        else:
            self.eye = glm.vec3(x, y, z)
        self.orientation = self._orientation_from_look(self.eye, self.center, up)
        self.update()

    def to_dict(self) -> dict:
        """导出当前 scene 状态为可 JSON 序列化的字典。"""
        return {
            "eye": [float(v) for v in self.eye],
            "center": [float(v) for v in self.center],
            "reference_up": [float(v) for v in self.reference_up],
            # 朝向用四元数 (w, x, y, z) 保存，避免欧拉角转换丢失 roll
            "orientation": [
                float(self.orientation.w),
                float(self.orientation.x),
                float(self.orientation.y),
                float(self.orientation.z),
            ],
            "near": float(self.near),
            "far": float(self.far),
            "fovy": float(self.fovy),
            "width": int(self.width),
            "height": int(self.height),
            "bgcolor": [float(v) for v in self.bgcolor],
            "viewmtx": self.viewmtx.tolist(),
            "projmtx": self.projmtx.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从 to_dict() 的结果恢复 scene 状态。"""
        scene = cls()
        if "eye" in data:
            scene.eye = glm.vec3(data["eye"])
        if "center" in data:
            scene.center = glm.vec3(data["center"])
        if "reference_up" in data:
            scene.reference_up = glm.vec3(data["reference_up"])
        if "orientation" in data:
            w, x, y, z = data["orientation"]
            scene.orientation = glm.normalize(glm.quat(w, x, y, z))
        else:
            scene.orientation = scene._orientation_from_look(
                scene.eye, scene.center, scene.reference_up
            )
        scene.near = float(data.get("near", scene.near))
        scene.far = float(data.get("far", scene.far))
        scene.fovy = float(data.get("fovy", scene.fovy))
        scene.width = int(data.get("width", scene.width))
        scene.height = int(data.get("height", scene.height))
        if "bgcolor" in data:
            scene.bgcolor = np.array(data["bgcolor"], dtype=np.float32)
        scene.update()
        # 矩阵若已导出则直接恢复，保留可能存在的自定义矩阵
        if "viewmtx" in data:
            scene.viewmtx = np.array(data["viewmtx"], dtype=np.float32).reshape(4, 4)
        if "projmtx" in data:
            scene.projmtx = np.array(data["projmtx"], dtype=np.float32).reshape(4, 4)
        return scene

    def to_file(self, path):
        """把当前 scene 状态写入 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        return path

    @classmethod
    def from_file(cls, path):
        """从 JSON 文件恢复 scene，返回新的 SplatScene。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __str__(self):
        s = f"""
SplatScene(
    bgcolor = {self.bgcolor},
    viewport = [{self.width}, {self.height}],
    eye = {self.eye},
    center = {self.center},
    up = {self.up},
    near = {self.near},
    far = {self.far},
    fovy = {self.fovy},
    viewmtx = {self.viewmtx},
    projmtx = {self.projmtx}
)
"""
        return s

class SplatSceneWidget:
    """SplatScene 的按钮 + 鼠标交互面板。

    4 个 rotate 按钮、4 个 pan 按钮、2 个 zoom 按钮，以及一个可绑定
    ipycanvas.Canvas 的鼠标操作层。

    注意：ipycanvas 的鼠标回调只给坐标，不提供 button，所以无法区分左键/中键。
    这里用 drag_mode 在“拖拽旋转”和“拖拽平移”之间切换；滚轮始终是 zoom。
    """

    def __init__(
        self,
        scene: SplatScene | None = None,
        rotate_step_deg: float = 5.0,
        pan_step: float = 0.05,
        zoom_step: float = 1.1,
    ):
        self.scene = scene if scene is not None else SplatScene()
        self.rotate_step = math.radians(rotate_step_deg)
        self.pan_step = pan_step
        self.zoom_step = zoom_step
        self._callbacks = []
        self._drag_last = None
        self.info = widgets.HTML()
        self._update_info()

        rotate_up = self._button("↑", self.rotate_up, "向上旋转")
        rotate_down = self._button("↓", self.rotate_down, "向下旋转")
        rotate_left = self._button("←", self.rotate_left, "向左旋转")
        rotate_right = self._button("→", self.rotate_right, "向右旋转")

        pan_up = self._button("↑", self.pan_up, "向上平移")
        pan_down = self._button("↓", self.pan_down, "向下平移")
        pan_left = self._button("←", self.pan_left, "向左平移")
        pan_right = self._button("→", self.pan_right, "向右平移")

        zoom_in = self._button("+", self.zoom_in, "zoom in")
        zoom_out = self._button("-", self.zoom_out, "zoom out")

        self.drag_mode = widgets.ToggleButtons(
            options=[("旋转", "rotate"), ("平移", "pan")],
            value="rotate",
            description="鼠标拖拽模式",
        )

        self.panel = widgets.VBox(
            [
                self._arrow_panel(
                    "rotate", rotate_up, rotate_down, rotate_left, rotate_right
                ),
                self._arrow_panel("pan", pan_up, pan_down, pan_left, pan_right),
                widgets.VBox(
                    [
                        widgets.HTML("<b>zoom</b>"),
                        widgets.HBox([zoom_in, zoom_out]),
                        self.drag_mode,
                    ]
                ),
                self.info,
            ]
        )

    @staticmethod
    def _button(label: str, callback, tooltip: str) -> widgets.Button:
        button = widgets.Button(
            description=label,
            tooltip=tooltip,
            layout=widgets.Layout(width="34px", padding="0px"),
        )
        button.on_click(lambda _: callback())
        return button

    @staticmethod
    def _arrow_panel(
        title: str,
        up: widgets.Button,
        down: widgets.Button,
        left: widgets.Button,
        right: widgets.Button,
    ) -> widgets.VBox:
        blank = widgets.Label("", layout=widgets.Layout(width="34px"))
        grid = widgets.GridBox(
            [blank, up, blank, left, down, right],
            layout=widgets.Layout(
                grid_template_columns="repeat(3, 34px)", grid_gap="2px"
            ),
        )
        return widgets.VBox([widgets.HTML(f"<b>{title}</b>"), grid])

    def on_change(self, callback):
        """注册场景变化后的回调（通常用来重绘 canvas）。"""
        self._callbacks.append(callback)
        return callback

    def _update_info(self):
        eye = self.scene.eye
        center = self.scene.center
        up = self.scene.up
        distance, head, pitch, roll = self.scene.camera_angles()
        self.info.value = (
            "<pre style='margin:0;font-family:monospace;"
            "font-size:11px;line-height:1.35'>"
            f"eye    : ({eye.x:8.3f}, {eye.y:8.3f}, {eye.z:8.3f})\n"
            f"center : ({center.x:8.3f}, {center.y:8.3f}, {center.z:8.3f})\n"
            f"up     : ({up.x:8.3f}, {up.y:8.3f}, {up.z:8.3f})\n"
            f"dist   : {distance:8.3f}\n"
            f"head   : {math.degrees(head) + 0.0:8.2f} deg\n"
            f"pitch  : {math.degrees(pitch) + 0.0:8.2f} deg\n"
            f"roll   : {math.degrees(roll) + 0.0:8.2f} deg\n"
            f"fovy   : {math.degrees(self.scene.fovy):8.2f} deg\n"
            f"near/far: {self.scene.near:g} / {self.scene.far:g}\n"
            f"viewport: {self.scene.width} x {self.scene.height}"
            "</pre>"
        )

    def _notify(self):
        self._update_info()
        for callback in list(self._callbacks):
            callback()

    def rotate_up(self):
        self.scene.camera_pitch_up(self.rotate_step)
        self._notify()

    def rotate_down(self):
        self.scene.camera_pitch_down(self.rotate_step)
        self._notify()

    def rotate_left(self):
        self.scene.camera_head_left(self.rotate_step)
        self._notify()

    def rotate_right(self):
        self.scene.camera_head_right(self.rotate_step)
        self._notify()

    def pan_up(self):
        self.scene.camera_pan_up(self.pan_step)
        self._notify()

    def pan_down(self):
        self.scene.camera_pan_down(self.pan_step)
        self._notify()

    def pan_left(self):
        self.scene.camera_pan_left(self.pan_step)
        self._notify()

    def pan_right(self):
        self.scene.camera_pan_right(self.pan_step)
        self._notify()

    def zoom_in(self):
        self.scene.camera_zoom(1.0 / self.zoom_step)
        self._notify()

    def zoom_out(self):
        self.scene.camera_zoom(self.zoom_step)
        self._notify()

    def bind_canvas(self, canvas):
        """绑定 canvas 的鼠标事件：拖拽按 drag_mode 旋转/平移，滚轮缩放。"""
        self._drag_last = None

        def on_mouse_down(x, y):
            self._drag_last = (x, y)

        def on_mouse_move(x, y):
            if self._drag_last is None:
                return

            last_x, last_y = self._drag_last
            dx = x - last_x
            dy = y - last_y
            self._drag_last = (x, y)
            if dx == 0 and dy == 0:
                return

            if self.drag_mode.value == "pan":
                self._drag_pan(canvas, dx, dy)
            else:
                self._drag_rotate(canvas, dx, dy)
            self._notify()

        def on_mouse_up(x, y):
            self._drag_last = None

        def on_mouse_out(x, y):
            self._drag_last = None

        def on_mouse_wheel(dx, dy):
            # ipycanvas 的 wheel 回调给的是 deltaX / deltaY。
            if dy == 0:
                return
            steps = max(-3.0, min(3.0, dy / 100.0))
            self.scene.camera_zoom(self.zoom_step**steps)
            self._notify()

        canvas.on_mouse_down(on_mouse_down)
        canvas.on_mouse_move(on_mouse_move)
        canvas.on_mouse_up(on_mouse_up)
        canvas.on_mouse_out(on_mouse_out)
        canvas.on_mouse_wheel(on_mouse_wheel)

    def _drag_rotate(self, canvas, dx: float, dy: float):
        # 横向拖过整个画布大约转一圈。
        speed = 2.0 * math.pi / max(canvas.width, 1)
        self.scene.camera_head_right(dx * speed)
        self.scene.camera_pitch_down(dy * speed)

    def _drag_pan(self, canvas, dx: float, dy: float):
        distance = glm.length(self.scene.eye - self.scene.center)
        world_per_pixel = (
            2.0 * distance * math.tan(self.scene.fovy * 0.5) / max(canvas.height, 1)
        )
        # 拖拽方向与内容移动方向一致。
        self.scene.camera_pan(-dx * world_per_pixel, dy * world_per_pixel)
