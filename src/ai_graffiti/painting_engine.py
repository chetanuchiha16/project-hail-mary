import cv2
import numpy as np
from contracts import GestureState
from config import CANVAS_SIZE, SPRAY_PARTICLES

class PaintingEngine:
    """
    Painting Engine.
    Responsibilities:
    - Canvas management using NumPy.
    - Spray brush implementation with Gaussian scatter.
    - AR merging of webcam feed and canvas.
    """
    def __init__(self):
        h, w = CANVAS_SIZE
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def update_canvas(self, state: GestureState):
        if not state.cursor_pos:
            return
            
        x, y = state.cursor_pos
        radius = 30
        
        if state.is_painting:
            xs = np.random.normal(x, radius/3, SPRAY_PARTICLES)
            ys = np.random.normal(y, radius/3, SPRAY_PARTICLES)
            for px, py in zip(xs, ys):
                px_int, py_int = int(px), int(py)
                h, w = CANVAS_SIZE
                if 0 <= px_int < w and 0 <= py_int < h:
                    # Draw a small dot for each particle
                    cv2.circle(self.canvas, (px_int, py_int), 1, state.brush_color, -1)
                    
        if state.is_erasing:
            # Clear a circular area for erasing
            cv2.circle(self.canvas, (int(x), int(y)), int(radius * 1.5), (0, 0, 0), -1)

    def render(self, webcam_frame):
        h, w = webcam_frame.shape[:2]
        canvas_resized = cv2.resize(self.canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(webcam_frame, 0.7, canvas_resized, 0.9, 0)

    def clear_canvas(self):
        self.canvas.fill(0)
