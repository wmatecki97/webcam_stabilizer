import cv2
import numpy as np
import logging

class FaceAligner:
    def __init__(self, max_eye_movement_threshold=100, logging_level=logging.DEBUG):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.first_left_eye_center = None
        self.prev_aligned_frame = None
        self.first_frame_processed = False
        self.prev_translation_matrix = None
        self.max_eye_movement_threshold = max_eye_movement_threshold
        self.prev_left_eye_center = None
        
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("FaceAligner initialized.")

    def detect_and_align(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        aligned_frame = frame
        if len(faces) == 0:
            logging.debug("No faces detected.")
            if self.prev_translation_matrix is not None:
                # Apply previous translation if available
                aligned_frame = cv2.warpAffine(frame, self.prev_translation_matrix, (frame.shape[1], frame.shape[0]))
                logging.debug("Applying previous translation.")
            elif self.prev_aligned_frame is not None:
                aligned_frame = self.prev_aligned_frame
            return aligned_frame, frame.shape[1], frame.shape[0]

        logging.debug(f"Detected {len(faces)} faces.")
        x, y, w, h = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)

        left_eye_center = None
        if len(eyes) >= 2:
            logging.debug(f"Detected {len(eyes)} eyes.")
            # Assuming the first two eyes are left and right
            eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
            eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]

            # Determine which eye is on the left
            if eye1_x < eye2_x:
                left_eye_x, left_eye_y, left_eye_w, left_eye_h = eye1_x, eye1_y, eye1_w, eye1_h
            else:
                left_eye_x, left_eye_y, left_eye_w, left_eye_h = eye2_x, eye2_y, eye2_w, eye2_h
            
            left_eye_center = (x + left_eye_x + left_eye_w // 2, y + left_eye_y + left_eye_h // 2)
            logging.debug(f"Left eye center: {left_eye_center}")

        if self.first_frame_processed:
            if self.first_left_eye_center is not None and left_eye_center is not None:
                
                if self.prev_left_eye_center is not None:
                    distance = np.sqrt((left_eye_center[0] - self.prev_left_eye_center[0])**2 + (left_eye_center[1] - self.prev_left_eye_center[1])**2)
                    if distance > self.max_eye_movement_threshold:
                        logging.warning(f"Eye movement distance {distance:.2f} exceeds threshold {self.max_eye_movement_threshold}. Ignoring frame.")
                        if self.prev_aligned_frame is not None:
                            aligned_frame = self.prev_aligned_frame
                        return aligned_frame, frame.shape[1], frame.shape[0]

                # Calculate the translation vector
                tx = self.first_left_eye_center[0] - left_eye_center[0]
                ty = self.first_left_eye_center[1] - left_eye_center[1]

                # Create the translation matrix
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

                # Apply the translation to the frame
                aligned_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
                logging.debug(f"Frame aligned with translation: {tx}, {ty}")
                self.prev_translation_matrix = translation_matrix
            elif self.prev_aligned_frame is not None:
                aligned_frame = self.prev_aligned_frame
        else:
            if left_eye_center is not None:
                self.first_left_eye_center = left_eye_center
                logging.debug(f"First left eye center set: {self.first_left_eye_center}")
            self.first_frame_processed = True
        
        self.prev_left_eye_center = left_eye_center
        self.prev_aligned_frame = aligned_frame
        return aligned_frame, frame.shape[1], frame.shape[0]
