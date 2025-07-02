import cv2
import face_recognition
import os
import numpy as np
import sys
import qrcode
import json
from datetime import datetime, timedelta

face_qr_mappings = {}

def show_success_message(frame, name):

    overlay = frame.copy()
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    border_size = 20
    cv2.rectangle(frame, (border_size, border_size), 
                 (frame.shape[1] - border_size, frame.shape[0] - border_size), 
                 (0, 255, 0), border_size)
    
    message = f"Welcome {name}!"
    sub_message = "Access Granted! Gate is Open!"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    (sub_text_width, sub_text_height), _ = cv2.getTextSize(sub_message, font, font_scale - 0.5, thickness)
    
    text_x = (frame.shape[1] - text_width) // 2
    text_y = frame.shape[0] // 2
    sub_text_x = (frame.shape[1] - sub_text_width) // 2
    sub_text_y = text_y + text_height + 50
    
    cv2.putText(frame, message, (text_x + 3, text_y + 3), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, sub_message, (sub_text_x + 3, sub_text_y + 3), font, font_scale - 0.5, (0, 0, 0), thickness + 2)
    
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, sub_message, (sub_text_x, sub_text_y), font, font_scale - 0.5, (0, 255, 0), thickness)

# to show notification on frame
def show_notification(frame, message, duration=30):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, message, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return duration

# to check if QR code needs to be regenerated
def should_regenerate_qr(qr_path):
    if not os.path.exists(qr_path):
        return True
    
    # the file's last modification time
    mod_time = os.path.getmtime(qr_path)
    mod_date = datetime.fromtimestamp(mod_time)
    current_date = datetime.now()
    
    # regenerate if the QR code is from a different day
    return mod_date.date() != current_date.date()

# to generate QR code for a face
def generate_qr_code(face_name):
    # a unique identifier for the face
    qr_data = {
        "face_name": face_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": str(np.random.randint(1000000))  # Add some randomness
    }
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code
    qr_path = os.path.join("submitted_faces", f"{face_name}_qr.png")
    qr_image.save(qr_path)
    return qr_path

# Load known faces from the "submitted_faces" folder
known_face_encodings = []
known_face_names = []

known_faces_dir = "submitted_faces"

# Check if the directory exists
if not os.path.exists(known_faces_dir):
    print(f"[ERROR] Directory '{known_faces_dir}' not found!")
    sys.exit(1)

# Load known faces
face_files = [f for f in os.listdir(known_faces_dir) if f.endswith((".jpg", ".png")) and not f.endswith("_qr.png")]
if not face_files:
    print(f"[ERROR] No face images found in '{known_faces_dir}'!")
    sys.exit(1)

for filename in face_files:
    image_path = os.path.join(known_faces_dir, filename)
    try:
        #convert to RGB
        image = face_recognition.load_image_file(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        if face_locations:
            # Get face encodings for the first face found
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                face_name = os.path.splitext(filename)[0]
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(face_name)
                
                # QR code if needed
                qr_path = os.path.join(known_faces_dir, f"{face_name}_qr.png")
                if should_regenerate_qr(qr_path):
                    qr_path = generate_qr_code(face_name)
                    print(f"[INFO] Generated new QR code for {face_name}")
                face_qr_mappings[face_name] = qr_path
        else:
            print(f"[WARNING] No face found in {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {str(e)}")

if not known_face_encodings:
    print("[ERROR] No valid faces could be loaded!")
    sys.exit(1)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Could not open webcam!")
    sys.exit(1)

print("[INFO] Starting webcam...")
print(f"[INFO] Loaded {len(known_face_names)} known faces")

last_face_detected = None
last_qr_detected = None
notification_timer = 0
notification_message = ""
verification_state = "FACE"  
qr_verification_start_time = None
QR_TIMEOUT_SECONDS = 15  
face_verification_start_time = None
FACE_VERIFICATION_SECONDS = 5  
success_timer = 0
SUCCESS_DISPLAY_SECONDS = 5  

qr_detector = cv2.QRCodeDetector()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Couldn't grab frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        current_face_detected = None
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    current_face_detected = name

            face_names.append(name)

        # face verification state
        if verification_state == "FACE":
            if current_face_detected:
                if current_face_detected != last_face_detected:
                    face_verification_start_time = datetime.now()
                    last_face_detected = current_face_detected
                    notification_message = f"Face detected: {current_face_detected}. Please hold still..."
                    notification_timer = 30
                elif face_verification_start_time:
                    elapsed_time = (datetime.now() - face_verification_start_time).total_seconds()
                    if elapsed_time >= FACE_VERIFICATION_SECONDS:
                        notification_message = f"Welcome {current_face_detected}!"
                        notification_timer = 30
                        verification_state = "QR"
                        qr_verification_start_time = datetime.now()
                        face_verification_start_time = None
            else:
                face_verification_start_time = None
                last_face_detected = None

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

        if verification_state == "SUCCESS":
            if success_timer <= 0:
                verification_state = "FACE"
                face_verification_start_time = None
                last_face_detected = None
                qr_verification_start_time = None
            else:
                success_timer -= 1
                show_success_message(frame, last_face_detected)

        elif verification_state == "QR":
            if qr_verification_start_time and (datetime.now() - qr_verification_start_time).total_seconds() > QR_TIMEOUT_SECONDS:
                notification_message = "QR Code verification timeout! Please try again."
                notification_timer = 30
                verification_state = "FACE"
                qr_verification_start_time = None
                face_verification_start_time = None
                last_face_detected = None
                continue

            try:
                retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(frame)
                current_qr_data = None
                
                if retval:
                    for i, data in enumerate(decoded_info):
                        if data:  
                            try:
                                qr_data = json.loads(data)
                                current_qr_data = qr_data
                                if points is not None and i < len(points):
                                    pts = points[i].astype(int)
                                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                            except:
                                continue

                if current_qr_data and last_face_detected:
                    if current_qr_data.get("face_name") == last_face_detected:
                        if notification_timer <= 0:
                            verification_state = "SUCCESS"
                            success_timer = SUCCESS_DISPLAY_SECONDS * 30  
                        last_qr_detected = current_qr_data.get("face_name")
                    else:
                        notification_message = "Error: Wrong QR Code! Please show the correct QR code."
                        notification_timer = 30
                        verification_state = "FACE"  
                        qr_verification_start_time = None
                        face_verification_start_time = None
                        last_face_detected = None
            except:
                pass

        if notification_timer > 0 and verification_state != "SUCCESS":
            notification_timer = show_notification(frame, notification_message, notification_timer)
            notification_timer -= 1

        state_text = f"State: {verification_state}"
        if verification_state == "QR" and qr_verification_start_time:
            remaining_time = QR_TIMEOUT_SECONDS - (datetime.now() - qr_verification_start_time).total_seconds()
            if remaining_time > 0:
                state_text += f" (Time remaining: {int(remaining_time)}s)"
        elif verification_state == "FACE" and face_verification_start_time:
            remaining_time = FACE_VERIFICATION_SECONDS - (datetime.now() - face_verification_start_time).total_seconds()
            if remaining_time > 0:
                state_text += f" (Hold still: {int(remaining_time)}s)"
        elif verification_state == "SUCCESS":
            state_text += f" (Success! Resetting in {int(success_timer/30)}s)"
        
        cv2.putText(frame, state_text, (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face and QR Code Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Program interrupted by user")
except Exception as e:
    print(f"[ERROR] An error occurred: {str(e)}")
finally:

    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Program ended")
