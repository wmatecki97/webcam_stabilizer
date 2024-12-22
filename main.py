import cv2
import os
from face_processing import process_frame

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full paths to the cascade files
    face_cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(script_dir, 'haarcascade_eye.xml')
    nose_cascade_path = os.path.join(script_dir, 'haarcascade_nose.xml')

    # Load the cascades
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

    # Check if the cascades were loaded successfully
    if face_cascade.empty():
        print(f"Error: Could not load face cascade file at {face_cascade_path}")
        return
    if eye_cascade.empty():
        print(f"Error: Could not load eye cascade file at {eye_cascade_path}")
        return
    if nose_cascade.empty():
        print(f"Error: Could not load nose cascade file at {nose_cascade_path}")
        return

    # Open the OBS virtual camera
    cap = cv2.VideoCapture(0)  # You might need to change the index (0, 1, 2, etc.)

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    prev_eye_distance = None
    initial_eye_positions = None
    initial_frame_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, prev_eye_distance, initial_eye_positions, initial_frame_shape = process_frame(frame, face_cascade, eye_cascade, nose_cascade, prev_eye_distance, initial_eye_positions, initial_frame_shape)

        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
