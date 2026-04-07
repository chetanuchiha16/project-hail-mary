import numpy as np
from .contracts import GestureState
from .config import PINCH_THRESHOLD, EMA_ALPHA

class GestureEngine:
    """
    Stub for the Gesture Engine.
    Responsibilities:
    - Hand detection and landmark extraction.
    - Gesture classification (pinch, open palm, closed fist).
    - Coordinate smoothing with EMA.
    """
    def __init__(self):
        self.alpha = EMA_ALPHA
        self.prev_pos = (0, 0)

    def process_frame(self, frame) -> GestureState:
        # Mocking logic for Day 1 integration
        return GestureState(
            cursor_pos=(640, 360),
            is_painting=False,
            is_erasing=False,
            brush_color=(255, 255, 255)
        )

    def _smooth_pos(self, raw_pos: tuple) -> tuple:
        # x_smooth = alpha * x_raw + (1 - alpha) * x_prev
        x_raw, y_raw = raw_pos
        x_prev, y_prev = self.prev_pos
        x_smooth = int(self.alpha * x_raw + (1 - self.alpha) * x_prev)
        y_smooth = int(self.alpha * y_raw + (1 - self.alpha) * y_prev)
        self.prev_pos = (x_smooth, y_smooth)
        return self.prev_pos
