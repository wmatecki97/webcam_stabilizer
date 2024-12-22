import cv2
import pyvirtualcam
import numpy as np
from image_processing import process_frame

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

    # Create a virtual camera
    try:
        with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps) as cam:
            print(f"Virtual camera started: {cam.device}")
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                frame_rgb = process_frame(frame)

                # Send the frame to the virtual camera
                cam.send(frame_rgb)
                cam.sleep_until_next_frame()

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
