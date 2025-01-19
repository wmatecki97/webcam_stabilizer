import cv2
import numpy as np
import logging
from cvzone.FaceMeshModule import FaceMeshDetector

class FaceAligner:
    def __init__(self, max_eye_movement_threshold=100, logging_level=logging.DEBUG, horizontal=50, vertical=50):
        self.fm = FaceMeshDetector(maxFaces=1)
        self.first_left_eye_center = None
        self.prev_aligned_frame = None
        self.first_frame_processed = False
        self.prev_translation_matrix = None
        self.max_eye_movement_threshold = max_eye_movement_threshold
        self.prev_left_eye_center = None
        self.left_eye_indices = [130,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
        self.horizontal = horizontal
        self.vertical = vertical
        
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

        # Calculate the target eye position based on config values
        target_x = int(frame.shape[1] * (self.horizontal / 100))
        target_y = int(frame.shape[0] * (self.vertical / 100))
        target_eye_center = np.array([target_x, target_y])

        if self.first_frame_processed:
            if target_eye_center is not None and left_eye_center is not None:

                # Calculate the translation vector
                tx = target_eye_center[0] - left_eye_center[0]
                ty = target_eye_center[1] - left_eye_center[1]

                # Create the translation matrix
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

                # Apply the translation to the frame
                aligned_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
                logging.debug(f"Frame aligned with translation: {tx}, {ty}")
                self.prev_translation_matrix = translation_matrix
            elif self.prev_aligned_frame is not None:
                aligned_frame = self.prev_aligned_frame
        else:
            self.first_frame_processed = True
        
        self.prev_left_eye_center = left_eye_center
        self.prev_aligned_frame = aligned_frame
        return aligned_frame, frame.shape[1], frame.shape[0]
