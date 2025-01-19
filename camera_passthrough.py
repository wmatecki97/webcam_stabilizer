import cv2
import pyvirtualcam
import numpy as np
from image_processing import process_frame
from face_detection import FaceAligner
import logging
import time
import os

def run_camera(config):
    # Open the default camera
    cap = cv2.VideoCapture(config.camera_index, config.camera_api)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    # Get camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Set default fps if camera doesn't provide it
    
    max_fps = config.max_fps
    if fps > max_fps:
        fps = max_fps

    logging.info(f"Camera opened with dimensions: {frame_width}x{frame_height} and fps: {fps}")


    face_aligner = FaceAligner(config = config)
    
    # Fixed output dimensions
    output_width = config.output_width
    output_height = config.output_height

    # Create a virtual camera with the fixed dimensions
    cam = pyvirtualcam.Camera(width=output_width, height=output_height, fps=fps)
    logging.info(f"Virtual camera started: {cam.device} with dimensions: {output_width}x{output_height}")
    
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            frame_start_time = time.time()
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Could not read frame from camera.")
                break
            logging.debug("Frame captured from camera.")

            # Resize the frame to 640x480
            frame = cv2.resize(frame, (640, 480))
            logging.debug("Frame resized to 640x480.")

            # Detect, align the frame
            aligned_frame, _, _ = face_aligner.detect_and_align(frame)
            logging.debug("Face detection and alignment complete.")

            # Process the frame (convert to RGB)
            frame_rgb = process_frame(aligned_frame)
            logging.debug("Frame processed (converted to RGB).")

            # Resize the processed frame to the virtual camera's dimensions
            frame_resized = cv2.resize(frame_rgb, (output_width, output_height))
            logging.debug(f"Frame resized to virtual camera dimensions: {output_width}x{output_height}")

            # Send the frame to the virtual camera
            cam.send(frame_resized)
            cam.sleep_until_next_frame()
            logging.debug("Frame sent to virtual camera.")
            
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_calc = frame_count / elapsed_time
                print(f"FPS: {fps_calc:.2f}")
                start_time = time.time()
                frame_count = 0
            
            elapsed_time = time.time() - frame_start_time
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
