# Face + QR Code Based Smart Home Security System

This project is a Python-based facial recognition and QR code verification system designed to enhance home security. It ensures that only authorized users with both a verified face and a daily-changing QR code can gain access (e.g., to unlock a door).

## How It Works

1. The camera detects a face in front of the door.
2. The system checks if the face matches a known user.
3. If matched, the user must present a QR code specific to that face (changes daily).
4. If both match, access is granted (e.g., the door unlocks via GPIO on Raspberry Pi).

## Features

- Real-time facial recognition
- QR code generation and decoding
- Dual-authentication: Face and QR must both match
- Live webcam feed using OpenCV
- Offline recognition with `face_recognition` and `dlib`
- Optimized for Raspberry Pi deployment

## QR Matching Logic

Each known user is associated with a QR code generated using:

A unique user ID
The current date (e.g., YYYY-MM-DD)
An optional salt or secret key
QR codes expire daily and must be presented within the valid day for access to be granted.
