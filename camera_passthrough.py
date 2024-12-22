import cv2
import pyvirtualcam
import numpy as np
from image_processing import process_frame
from face_detection import FaceAligner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    # Get camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Set default fps if camera doesn't provide it

    logging.info(f"Camera opened with dimensions: {frame_width}x{frame_height} and fps: {fps}")

    face_aligner = FaceAligner()

    # Create a virtual camera with original dimensions
    cam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps)
    logging.info(f"Virtual camera started: {cam.device}")
    
    first_frame = True
    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Could not read frame from camera.")
                break
            logging.debug("Frame captured from camera.")

            # Detect, align and crop the frame
            aligned_frame, cropped_width, cropped_height = face_aligner.detect_and_align(frame)
            logging.debug(f"Face detection and alignment complete. Cropped dimensions: {cropped_width}x{cropped_height}")

            # Process the frame (convert to RGB)
            frame_rgb = process_frame(aligned_frame)
            logging.debug("Frame processed (converted to RGB).")

            # Re-initialize the virtual camera with the correct dimensions after the first frame
            if first_frame:
                first_frame = False
                cam.close()
                cam = pyvirtualcam.Camera(width=cropped_width, height=cropped_height, fps=fps)
                logging.info(f"Virtual camera re-initialized: {cam.device} with dimensions: {cropped_width}x{cropped_height}")

            # Send the frame to the virtual camera
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()
            logging.debug("Frame sent to virtual camera.")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting program.")
                break
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Release the camera and destroy all windows
        if cap:
            cap.release()
            logging.info("Camera released.")
        if cam:
            cam.close()
            logging.info("Virtual camera closed.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
