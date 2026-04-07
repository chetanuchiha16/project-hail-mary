import cv2
import numpy as np
from .contracts import GestureState
from .config import CANVAS_SIZE, SPRAY_PARTICLES

class PaintingEngine:
    """
    Stub for the Painting Engine.
    Responsibilities:
    - Canvas management using NumPy.
    - Spray brush implementation with Gaussian scatter.
    - AR merging of webcam feed and canvas.
    """
    def __init__(self):
        h, w = CANVAS_SIZE
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def update_canvas(self, state: GestureState):
        if state.is_painting:
            # np.random.normal(x, radius/3, N)
            pass

    def render(self, webcam_frame):
        # cv2.addWeighted(webcam_frame, 0.7, canvas_bgr, 0.9, 0)
        return cv2.addWeighted(webcam_frame, 0.7, self.canvas, 0.9, 0)
