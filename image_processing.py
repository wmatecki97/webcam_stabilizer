import cv2
import numpy as np

def process_frame(frame):
    """
    Processes a single frame by cropping 10% from each side and converting to RGB.

    Args:
        frame: A BGR frame from OpenCV.

    Returns:
        A cropped RGB frame.
    """
    height, width = frame.shape[:2]
    
    # Calculate the crop amount
    crop_x = int(width * 0.1)
    crop_y = int(height * 0.1)

    # Crop the frame
    cropped_frame = frame[crop_y:height-crop_y, crop_x:width-crop_x]

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    return frame_rgb
