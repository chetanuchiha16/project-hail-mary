import cv2
import math
import numpy as np
import mediapipe as mp
from contracts import GestureState
from config import PINCH_THRESHOLD, EMA_ALPHA, HAND_MODEL_COMPLEXITY, CANVAS_SIZE

class GestureEngine:
    """
    Gesture Engine.
    Responsibilities:
    - Hand detection and landmark extraction using MediaPipe.
    - Gesture classification (pinch for paint, closed fist for erase).
    - Coordinate smoothing with EMA.
    """
    def __init__(self):
        self.alpha = EMA_ALPHA
        self.prev_pos = (0, 0)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=HAND_MODEL_COMPLEXITY,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        self.canvas_h, self.canvas_w = CANVAS_SIZE

    def process_frame(self, frame) -> GestureState:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        raw_pos = self.prev_pos
        is_painting = False
        is_erasing = False
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            raw_x = int(index_tip.x * self.canvas_w)
            raw_y = int(index_tip.y * self.canvas_h)
            raw_pos = (raw_x, raw_y)
            
            # Calculate distance for pinch (painting)
            pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            
            # Simple closed fist logic: distance from middle finger tip to wrist is low
            fist_dist = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            
            if pinch_dist < PINCH_THRESHOLD:
                is_painting = True
            elif fist_dist < 0.25: # Arbitrary threshold for a closed fist (erasing)
                is_erasing = True

        cursor_pos = self._smooth_pos(raw_pos)

        return GestureState(
            cursor_pos=cursor_pos,
            is_painting=is_painting,
            is_erasing=is_erasing,
            brush_color=(255, 255, 255)
        )

    def _smooth_pos(self, raw_pos: tuple) -> tuple:
        x_raw, y_raw = raw_pos
        x_prev, y_prev = self.prev_pos
        x_smooth = int(self.alpha * x_raw + (1 - self.alpha) * x_prev)
        y_smooth = int(self.alpha * y_raw + (1 - self.alpha) * y_prev)
        self.prev_pos = (x_smooth, y_smooth)
        return self.prev_pos
