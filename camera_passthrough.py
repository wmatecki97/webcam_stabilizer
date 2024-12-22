import cv2
import pyvirtualcam
import numpy as np
from image_processing import process_frame
from face_detection import FaceAligner

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Set default fps if camera doesn't provide it

    face_aligner = FaceAligner()

    # Create a virtual camera
    try:
        # Initialize with dummy values, will be updated after first frame
        cam = None
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            # Detect, align and crop the frame
            aligned_frame, cropped_width, cropped_height = face_aligner.detect_and_align(frame)

            # Process the frame (convert to RGB)
            frame_rgb = process_frame(aligned_frame)

            # Initialize the virtual camera with the correct dimensions
            if cam is None:
                cam = pyvirtualcam.Camera(width=cropped_width, height=cropped_height, fps=fps)
                print(f"Virtual camera started: {cam.device}")

            # Send the frame to the virtual camera
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Release the camera and destroy all windows
        if cap:
            cap.release()
        if cam:
            cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
