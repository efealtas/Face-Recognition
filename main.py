import face_recognition
import os

known_face_encodings = []
known_face_names = []

known_dir = 'submitted_faces'

for filename in os.listdir(known_dir):
    path = os.path.join(known_dir, filename)
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]  # assumes one face per image
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])  # use file name as person name

import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert color (OpenCV uses BGR)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces & encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Scale back face location to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box & label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

