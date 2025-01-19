import cv2
import numpy as np
import logging
from cvzone.FaceMeshModule import FaceMeshDetector

class FaceAligner:
    def __init__(self, max_eye_movement_threshold=100, logging_level=logging.DEBUG):
        self.fm = FaceMeshDetector(maxFaces=1)
        self.first_left_eye_center = None
        self.prev_aligned_frame = None
        self.first_frame_processed = False
        self.prev_translation_matrix = None
        self.max_eye_movement_threshold = max_eye_movement_threshold
        self.prev_left_eye_center = None
        self.left_eye_indices = [130,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
        
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("FaceAligner initialized.")

    def detect_and_align(self, frame):
        frame, faces = self.fm.findFaceMesh(frame, draw=False)
        aligned_frame = frame
        
        if not faces:
            logging.debug("No faces detected.")
            if self.prev_translation_matrix is not None:
                # Apply previous translation if available
                aligned_frame = cv2.warpAffine(frame, self.prev_translation_matrix, (frame.shape[1], frame.shape[0]))
                logging.debug("Applying previous translation.")
            elif self.prev_aligned_frame is not None:
                aligned_frame = self.prev_aligned_frame
            return aligned_frame, frame.shape[1], frame.shape[0]

        logging.debug(f"Detected {len(faces)} faces.")
        face = faces[0]
        left_eye_points = np.array([face[index] for index in self.left_eye_indices])
        left_eye_center = np.mean(left_eye_points[:,:2], axis=0, dtype=np.int32)
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
