import numpy as np
import torch
import cv2
from typing import Optional, Dict, Any


class VideoDisplay:
    def __init__(self, window_name="HunyuanWorld Interactive", keyboard_overlay=True):
        self.window_name = window_name
        self.keyboard_overlay = keyboard_overlay
        self.current_actions = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}
        self.frame_buffer = []
        self.is_running = False

    def update_actions(self, actions: Dict[str, int]):
        self.current_actions = actions

    def tensor_to_display_frame(self, frame: torch.Tensor) -> np.ndarray:
        if frame.ndim == 5:
            frame = frame[0]
        if frame.ndim == 4:
            frame = frame[0]

        frame = frame.permute(1, 2, 0)

        frame = frame.cpu().float().numpy()

        frame = np.clip(frame, 0, 1)

        frame = (frame * 255).astype(np.uint8)

        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def draw_keyboard_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.keyboard_overlay:
            return frame

        h, w = frame.shape[:2]

        key_size = 50
        key_spacing = 4
        margin = 20

        wasd_width = 3 * key_size + 2 * key_spacing
        wasd_height = 2 * key_size + key_spacing

        overlay = np.zeros((wasd_height + margin * 2, wasd_width + margin * 2, 4), dtype=np.uint8)

        def draw_key(x, y, label, active):
            color = (30, 120, 255) if active else (50, 50, 50)
            cv2.rectangle(overlay, (x + 2, y + 2), (x + key_size - 2, y + key_size - 2), color, -1)
            cv2.rectangle(overlay, (x + 2, y + 2), (x + key_size - 2, y + key_size - 2), (200, 200, 200), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            text_x = x + (key_size - text_size[0]) // 2
            text_y = y + (key_size + text_size[1]) // 2
            cv2.putText(overlay, label, (text_x, text_y), font, 0.7, (255, 255, 255), 2)

        forward_val = self.current_actions.get("forward", 0)
        left_val = self.current_actions.get("left", 0)
        yaw_val = self.current_actions.get("yaw", 0)
        pitch_val = self.current_actions.get("pitch", 0)

        w_active = forward_val > 0
        s_active = forward_val < 0
        a_active = left_val > 0
        d_active = left_val < 0

        keys = [
            ("W", 1, 0, w_active),
            ("A", 0, 1, a_active),
            ("S", 1, 1, s_active),
            ("D", 2, 1, d_active),
        ]

        for label, col, row, is_active in keys:
            x = margin + col * (key_size + key_spacing)
            y = margin + row * (key_size + key_spacing)
            draw_key(x, y, label, is_active)

        arrow_width = 3 * key_size + 2 * key_spacing
        arrow_height = 2 * key_size + key_spacing
        arrow_overlay = np.zeros((arrow_height + margin * 2, arrow_width + margin * 2, 4), dtype=np.uint8)

        def draw_arrow(x, y, direction, active):
            color = (30, 120, 255) if active else (50, 50, 50)
            cv2.rectangle(arrow_overlay, (x + 2, y + 2), (x + key_size - 2, y + key_size - 2), color, -1)
            cv2.rectangle(arrow_overlay, (x + 2, y + 2), (x + key_size - 2, y + key_size - 2), (200, 200, 200), 1)

            cx, cy = x + key_size // 2, y + key_size // 2
            size = key_size // 4

            if direction == "up":
                pts = np.array([[cx, cy - size], [cx - size, cy + size // 2], [cx + size, cy + size // 2]], np.int32)
            elif direction == "down":
                pts = np.array([[cx, cy + size], [cx - size, cy - size // 2], [cx + size, cy - size // 2]], np.int32)
            elif direction == "left":
                pts = np.array([[cx - size, cy], [cx + size // 2, cy - size], [cx + size // 2, cy + size]], np.int32)
            elif direction == "right":
                pts = np.array([[cx + size, cy], [cx - size // 2, cy - size], [cx - size // 2, cy + size]], np.int32)

            cv2.fillPoly(arrow_overlay, [pts], (255, 255, 255))

        up_active = pitch_val > 0
        down_active = pitch_val < 0
        left_active = yaw_val < 0
        right_active = yaw_val > 0

        arrows = [
            ("up", 1, 0, up_active),
            ("left", 0, 1, left_active),
            ("down", 1, 1, down_active),
            ("right", 2, 1, right_active),
        ]

        for direction, col, row, is_active in arrows:
            x = margin + col * (key_size + key_spacing)
            y = margin + row * (key_size + key_spacing)
            draw_arrow(x, y, direction, is_active)

        wasd_x = margin
        wasd_y = h - wasd_height - margin * 3

        if wasd_y >= 0 and wasd_x + wasd_width + margin * 2 <= w:
            overlay_resized = cv2.resize(overlay, (wasd_width + margin * 2, wasd_height + margin * 2))
            overlay_alpha = overlay_resized[:, :, 3:4].astype(np.float32) / 255.0
            overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)

            y1, y2 = wasd_y, min(wasd_y + arrow_height + margin * 2, h)
            x1, x2 = wasd_x, min(wasd_x + wasd_width + margin * 2, w)

            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2] = (
                    overlay_rgb[:y2-y1, :x2-x1] * overlay_alpha[:y2-y1, :x2-x1] +
                    frame[y1:y2, x1:x2].astype(np.float32) * (1 - overlay_alpha[:y2-y1, :x2-x1])
                ).astype(np.uint8)

        arrow_x = w - arrow_width - margin * 3
        arrow_y = h - arrow_height - margin * 3

        if arrow_y >= 0 and arrow_x >= 0:
            arrow_resized = cv2.resize(arrow_overlay, (arrow_width + margin * 2, arrow_height + margin * 2))
            arrow_alpha = arrow_resized[:, :, 3:4].astype(np.float32) / 255.0
            arrow_rgb = arrow_resized[:, :, :3].astype(np.float32)

            y1, y2 = arrow_y, min(arrow_y + arrow_height + margin * 2, h)
            x1, x2 = arrow_x, min(arrow_x + arrow_width + margin * 2, w)

            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2] = (
                    arrow_rgb[:y2-y1, :x2-x1] * arrow_alpha[:y2-y1, :x2-x1] +
                    frame[y1:y2, x1:x2].astype(np.float32) * (1 - arrow_alpha[:y2-y1, :x2-x1])
                ).astype(np.uint8)

        return frame

    def display_frame(self, frame: torch.Tensor, actions: Optional[Dict[str, int]] = None):
        if actions is not None:
            self.update_actions(actions)

        display_frame = self.tensor_to_display_frame(frame)

        if self.keyboard_overlay:
            display_frame = self.draw_keyboard_overlay(display_frame)

        cv2.imshow(self.window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        cv2.destroyWindow(self.window_name)


_display_instance = None


def get_display(window_name="HunyuanWorld Interactive", create=False, **kwargs):
    global _display_instance
    if create:
        _display_instance = VideoDisplay(window_name=window_name, **kwargs)
    return _display_instance


def close_display():
    global _display_instance
    if _display_instance is not None:
        _display_instance.close()
        _display_instance = None


def display_frame(frame: torch.Tensor, actions: Optional[Dict[str, int]] = None):
    global _display_instance
    if _display_instance is None:
        _display_instance = VideoDisplay()
    return _display_instance.display_frame(frame, actions)


def parse_keyboard_input(key: int) -> tuple:
    key_mappings = {
        ord('w'): ("forward", 1),
        ord('s'): ("forward", -1),
        ord('a'): ("left", 1),
        ord('d'): ("left", -1),
        82: ("yaw", 1),
        83: ("yaw", -1),
        84: ("pitch", 1),
        85: ("pitch", -1),
    }

    actions = {"forward": 0, "left": 0, "yaw": 0, "pitch": 0}

    pressed = key != -1 and key != 255

    if key == ord('w'):
        actions["forward"] = 1
    elif key == ord('s'):
        actions["forward"] = -1
    elif key == ord('a'):
        actions["left"] = 1
    elif key == ord('d'):
        actions["left"] = -1
    elif key == 82:
        actions["yaw"] = 1
    elif key == 83:
        actions["yaw"] = -1
    elif key == 84:
        actions["pitch"] = 1
    elif key == 85:
        actions["pitch"] = -1

    return pressed, actions
