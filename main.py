import cv2
from face_processing import process_frame

def main():
    # Load the cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_nose.xml')

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
