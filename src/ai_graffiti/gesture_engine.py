import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from contracts import GestureState
from config import PINCH_THRESHOLD, EMA_ALPHA, CANVAS_SIZE, HAND_MODEL_PATH


class GestureEngine:
    """
    Gesture Engine.
    Responsibilities:
    - Hand detection and landmark extraction using MediaPipe Tasks API.
    - Gesture classification (pinch for paint, closed fist for erase).
    - Coordinate smoothing with EMA.
    """

    # Landmark indices (same as the old mp.solutions.hands constants)
    INDEX_FINGER_TIP = 8
    THUMB_TIP = 4
    MIDDLE_FINGER_TIP = 12
    WRIST = 0

    def __init__(self):
        self.alpha = EMA_ALPHA
        self.prev_pos = (0, 0)
        self.canvas_h, self.canvas_w = CANVAS_SIZE

        # Resolve the model path relative to this file
        model_path = os.path.join(os.path.dirname(__file__), HAND_MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, frame) -> GestureState:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamps must be monotonically increasing for VIDEO mode
        self._frame_timestamp_ms += 33  # ~30 FPS
        results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        raw_pos = self.prev_pos
        is_painting = False
        is_erasing = False

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]

            index_tip = landmarks[self.INDEX_FINGER_TIP]
            thumb_tip = landmarks[self.THUMB_TIP]
            middle_tip = landmarks[self.MIDDLE_FINGER_TIP]
            wrist = landmarks[self.WRIST]

            raw_x = int(index_tip.x * self.canvas_w)
            raw_y = int(index_tip.y * self.canvas_h)
            raw_pos = (raw_x, raw_y)

            # Calculate distance for pinch (painting)
            pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            # Simple closed fist logic: distance from middle finger tip to wrist is low
            fist_dist = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)

            if pinch_dist < PINCH_THRESHOLD:
                is_painting = True
            elif fist_dist < 0.25:  # Arbitrary threshold for a closed fist (erasing)
                is_erasing = True

        cursor_pos = self._smooth_pos(raw_pos)

        return GestureState(
            cursor_pos=cursor_pos,
            is_painting=is_painting,
            is_erasing=is_erasing,
            brush_color=(255, 255, 255),
        )

    def _smooth_pos(self, raw_pos: tuple) -> tuple:
        x_raw, y_raw = raw_pos
        x_prev, y_prev = self.prev_pos
        x_smooth = int(self.alpha * x_raw + (1 - self.alpha) * x_prev)
        y_smooth = int(self.alpha * y_raw + (1 - self.alpha) * y_prev)
        self.prev_pos = (x_smooth, y_smooth)
        return self.prev_pos

    def close(self):
        """Release the landmarker resources."""
        self.landmarker.close()
