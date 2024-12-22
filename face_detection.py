import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceAligner:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.prev_left_eye_centers = []
        self.prev_aligned_frame = None
        self.frame_counter = 0
        logging.info("FaceAligner initialized.")

    def detect_and_align(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, output_reject_levels=True)
        
        if len(faces) == 0:
            logging.debug("No faces detected.")
            if self.prev_aligned_frame is not None:
                return self.prev_aligned_frame, frame.shape[1], frame.shape[0]
            else:
                return frame, frame.shape[1], frame.shape[0]
        
        detected_faces, _, detection_probabilities = faces
        
        if len(detected_faces) == 0 or max(detection_probabilities) < 0.8:
            logging.debug(f"Low face detection probability: {max(detection_probabilities) if detection_probabilities.size > 0 else 'N/A'}. Skipping alignment.")
            if self.prev_aligned_frame is not None:
                return self.prev_aligned_frame, frame.shape[1], frame.shape[0]
            else:
                return frame, frame.shape[1], frame.shape[0]

        logging.debug(f"Detected {len(detected_faces)} faces.")
        x, y, w, h = detected_faces[0]
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

        aligned_frame = frame
        if left_eye_center is not None:
            self.prev_left_eye_centers.append(left_eye_center)
            if len(self.prev_left_eye_centers) > 30:
                self.prev_left_eye_centers.pop(0)
            
            if len(self.prev_left_eye_centers) > 0:
                avg_left_eye_x = np.mean([center[0] for center in self.prev_left_eye_centers])
                avg_left_eye_y = np.mean([center[1] for center in self.prev_left_eye_centers])
                avg_left_eye_center = (int(avg_left_eye_x), int(avg_left_eye_y))

                if self.prev_left_eye_centers and self.frame_counter > 0:
                    # Calculate the translation vector
                    tx = self.prev_left_eye_centers[-1][0] - avg_left_eye_center[0]
                    ty = self.prev_left_eye_centers[-1][1] - avg_left_eye_center[1]

                    # Create the translation matrix
                    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

                    # Apply the translation to the frame
                    aligned_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
                    logging.debug(f"Frame aligned with translation: {tx}, {ty}")
        
        self.prev_aligned_frame = aligned_frame
        self.frame_counter += 1
        return aligned_frame, frame.shape[1], frame.shape[0]
