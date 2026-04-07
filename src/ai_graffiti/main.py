import cv2
import time
from config import HAND_MODEL_COMPLEXITY
from gesture_engine import GestureEngine
from painting_engine import PaintingEngine

class MainApp:
    """
    Stub for the Main Application Loop.
    Responsibilities:
    - Orchestrating the data flow: Camera -> GestureEngine -> PaintingEngine -> Display.
    - FPS tracking and HUD overlay.
    - Global parameter tuning via config.py.
    """
    def __init__(self):
        self.gesture_engine = GestureEngine()
        self.painting_engine = PaintingEngine()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        if not self.cap.isOpened():
            print("Error: Camera initialization failed.")
            return

        prev_time = time.time()

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Error: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            # Orchestration logic
            state = self.gesture_engine.process_frame(frame)
            self.painting_engine.update_canvas(state)
            combined_frame = self.painting_engine.render(frame)

            # HUD implementation here
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            mode = "IDLE"
            if state.is_painting:
                mode = "PAINTING"
            elif state.is_erasing:
                mode = "ERASING"
                
            # Mode
            cv2.putText(combined_frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # FPS
            cv2.putText(combined_frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Cursor
            cv2.circle(combined_frame, state.cursor_pos, 5, (0, 0, 255), -1)

            cv2.imshow("AI Graffiti Wall", combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
            elif key == ord('c') or key == ord('C'):
                self.painting_engine.clear_canvas()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MainApp()
    app.run()
