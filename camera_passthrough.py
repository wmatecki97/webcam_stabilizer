import cv2
import pyvirtualcam
import numpy as np
from image_processing import process_frame
from face_detection import FaceAligner
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    # Get camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Set default fps if camera doesn't provide it
    
    max_fps = 30
    if fps > max_fps:
        fps = max_fps

    logging.info(f"Camera opened with dimensions: {frame_width}x{frame_height} and fps: {fps}")

    face_aligner = FaceAligner()
    
    # Fixed output dimensions
    output_width = 1280
    output_height = 720

    # Create a virtual camera with the fixed dimensions
    cam = pyvirtualcam.Camera(width=output_width, height=output_height, fps=fps)
    logging.info(f"Virtual camera started: {cam.device} with dimensions: {output_width}x{output_height}")
    
    try:
        while True:
            start_time = time.time()
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Could not read frame from camera.")
                break
            logging.debug("Frame captured from camera.")

            # Detect, align the frame
            aligned_frame, _, _ = face_aligner.detect_and_align(frame)
            logging.debug("Face detection and alignment complete.")

            # Process the frame (convert to RGB)
            frame_rgb = process_frame(aligned_frame)
            logging.debug("Frame processed (converted to RGB).")

            # Send the frame to the virtual camera
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()
            logging.debug("Frame sent to virtual camera.")
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1/fps) - elapsed_time)
            time.sleep(sleep_time)

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
