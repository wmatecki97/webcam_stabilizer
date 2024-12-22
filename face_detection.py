import cv2

def detect_features(frame, face_cascade, eye_cascade, nose_cascade):
    """Detects faces, eyes, and nose in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = []
    nose = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        noses = nose_cascade.detectMultiScale(roi_gray)
        if len(noses) > 0:
            nose = (noses[0][0] + x + noses[0][2] // 2, noses[0][1] + y + noses[0][3] // 2)
        
        eyes = [(ex + x + ew // 2, ey + y + eh // 2) for (ex, ey, ew, eh) in eyes]
        break # Consider only the first detected face

    return eyes, nose
