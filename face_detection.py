import cv2
import numpy as np

class FaceAligner:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.prev_left_eye_center = None

    def detect_and_align(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return frame, frame.shape[1], frame.shape[0]

        x, y, w, h = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)

        left_eye_center = None
        if len(eyes) >= 2:
            # Assuming the first two eyes are left and right
            eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
            eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]

            # Determine which eye is on the left
            if eye1_x < eye2_x:
                left_eye_x, left_eye_y, left_eye_w, left_eye_h = eye1_x, eye1_y, eye1_w, eye1_h
            else:
                left_eye_x, left_eye_y, left_eye_w, left_eye_h = eye2_x, eye2_y, eye2_w, eye2_h
            
            left_eye_center = (x + left_eye_x + left_eye_w // 2, y + left_eye_y + left_eye_h // 2)

        aligned_frame = frame
        if self.prev_left_eye_center is not None and left_eye_center is not None:
            # Calculate the translation vector
            tx = self.prev_left_eye_center[0] - left_eye_center[0]
            ty = self.prev_left_eye_center[1] - left_eye_center[1]

            # Create the translation matrix
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

            # Apply the translation to the frame
            aligned_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))

        # Update previous left eye position
        if left_eye_center is not None:
            self.prev_left_eye_center = left_eye_center

        # Calculate crop after alignment
        height, width = aligned_frame.shape[:2]
        crop_x = int(width * 0.1)
        crop_y = int(height * 0.1)

        # Crop the frame
        cropped_frame = aligned_frame[crop_y:height-crop_y, crop_x:width-crop_x]

        return cropped_frame, cropped_frame.shape[1], cropped_frame.shape[0]
