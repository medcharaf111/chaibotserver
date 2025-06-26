import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface.models.spoofing.FasNet import Fasnet

if len(sys.argv) < 2:
    print('Usage: python video_spoof_check.py <video_file>')
    sys.exit(1)

video_path = sys.argv[1]

# Initialize face and liveness detector
face_app = FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=0, det_size=(640, 640))
liveness_detector = Fasnet()

cap = cv2.VideoCapture(video_path)
spoof_detected = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        is_live, live_score = liveness_detector.analyze(frame, (x1, y1, w, h))
        if not is_live:
            spoof_detected = True
            break
    if spoof_detected:
        break
cap.release()

if spoof_detected:
    print('spoof detected')
    sys.exit(2)
else:
    print('no spoof detected')
    sys.exit(0) 