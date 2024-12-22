import cv2
import numpy as np
import math
from face_detection import detect_features

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def process_frame(frame, face_cascade, eye_cascade, nose_cascade, prev_eye_distance, initial_eye_positions, initial_frame_shape):
    """Processes a single frame."""
    eyes, nose = detect_features(frame, face_cascade, eye_cascade, nose_cascade)
    
    if len(eyes) != 2:
        return frame, prev_eye_distance, initial_eye_positions, initial_frame_shape

    current_eye_distance = calculate_distance(eyes[0], eyes[1])
    
    if prev_eye_distance is None:
        # First frame, initialize
        initial_eye_positions = eyes
        initial_frame_shape = frame.shape[:2]
        return frame, current_eye_distance, initial_eye_positions, initial_frame_shape

    if abs(current_eye_distance - prev_eye_distance) > 10: # Threshold for change
        scale_factor = prev_eye_distance / current_eye_distance
        
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate the shift needed to keep eyes in the same position
        resized_eyes = [(int(eye[0] * scale_factor), int(eye[1] * scale_factor)) for eye in eyes]
        
        shift_x = initial_eye_positions[0][0] - resized_eyes[0][0]
        shift_y = initial_eye_positions[0][1] - resized_eyes[0][1]
        
        # Create a translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply the translation
        shifted_frame = cv2.warpAffine(resized_frame, translation_matrix, (resized_frame.shape[1], resized_frame.shape[0]))
        
        # Crop the image to the original size
        x = max(0, int(shift_x))
        y = max(0, int(shift_y))
        
        cropped_frame = shifted_frame[y:y+initial_frame_shape[0], x:x+initial_frame_shape[1]]
        
        # Handle cases where the cropped frame is smaller than the original
        if cropped_frame.shape[0] != initial_frame_shape[0] or cropped_frame.shape[1] != initial_frame_shape[1]:
            
            final_frame = np.zeros((initial_frame_shape[0], initial_frame_shape[1], 3), dtype=np.uint8)
            
            h_crop = min(cropped_frame.shape[0], initial_frame_shape[0])
            w_crop = min(cropped_frame.shape[1], initial_frame_shape[1])
            
            final_frame[:h_crop, :w_crop] = cropped_frame[:h_crop, :w_crop]
            
            return final_frame, current_eye_distance, initial_eye_positions, initial_frame_shape
        
        return cropped_frame, current_eye_distance, initial_eye_positions, initial_frame_shape
    
    return frame, current_eye_distance, initial_eye_positions, initial_frame_shape
