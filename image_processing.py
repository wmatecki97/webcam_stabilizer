import cv2

def process_frame(frame):
    """
    Processes a single frame.

    Args:
        frame: A BGR frame from OpenCV.

    Returns:
        A RGB frame.
    """
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb
