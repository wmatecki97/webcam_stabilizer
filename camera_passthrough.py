import cv2
import pyvirtualcam
import numpy as np

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

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Send the frame to the virtual camera
                cam.send(frame_rgb)
                cam.sleep_until_next_frame()

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except pyvirtualcam.PyVirtualCamException as e:
        print(f"Error creating virtual camera: {e}")

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
