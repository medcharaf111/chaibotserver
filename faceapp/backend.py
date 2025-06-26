import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from threading import Thread, Lock
import time
from insightface.app import FaceAnalysis
from deepface.models.spoofing.FasNet import Fasnet
import logging
from flask_cors import CORS
import tempfile
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
import zipfile
import uuid
import threading

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_backend")

# Global face recognizer and liveness detector
class VideoFaceRecognizer:
    def __init__(self, model_name='buffalo_s', images_folder="images"):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.liveness_detector = Fasnet()
        self.images_folder = images_folder
        self.known_faces = {}
        self.known_face_embeddings = []
        self.known_face_names = []
        self.load_known_faces(images_folder)
        self.lock = Lock()

    def load_known_faces(self, images_folder="images"):
        import os
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.known_faces = {}
        self.known_face_embeddings = []
        self.known_face_names = []
        for filename in os.listdir(images_folder):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_path = os.path.join(images_folder, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.app.get(rgb_image)
                if len(faces) > 0:
                    face = faces[0]
                    embedding = face.embedding
                    person_name = os.path.splitext(filename)[0]
                    self.known_faces[person_name] = {
                        'embedding': embedding,
                        'image_path': image_path
                    }
                    self.known_face_embeddings.append(embedding)
                    self.known_face_names.append(person_name)

    def recognize_face(self, face_embedding, threshold=0.4):
        if len(self.known_face_embeddings) == 0:
            return None, 0
        similarities = []
        for known_embedding in self.known_face_embeddings:
            similarity = np.dot(face_embedding, known_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
            )
            similarities.append(similarity)
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        if best_similarity >= threshold:
            return self.known_face_names[best_match_idx], best_similarity
        else:
            return "Unknown", best_similarity

    def detect_and_recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)
        recognition_results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            is_live, live_score = self.liveness_detector.analyze(frame, (x1, y1, w, h))
            if is_live:
                if hasattr(face, 'embedding'):
                    person_name, confidence = self.recognize_face(face.embedding)
                else:
                    person_name, confidence = 'Unknown', 0
            else:
                person_name, confidence = 'Spoof', 0
            recognition_results.append({
                'face': face,
                'name': person_name,
                'confidence': confidence,
                'is_live': is_live,
                'live_score': live_score
            })
        return recognition_results

    def draw_faces(self, frame, recognition_results):
        for result in recognition_results:
            face = result['face']
            name = result['name']
            confidence = result['confidence']
            is_live = result.get('is_live', None)
            live_score = result.get('live_score', None)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            if name == "Spoof" or not is_live:
                color = (0, 165, 255)
            elif name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if is_live is not None:
                liveness_label = f"Live: {is_live} ({live_score:.2f})"
            else:
                liveness_label = "Live: N/A"
            label = f"{name} ({confidence:.2f}) | {liveness_label}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if hasattr(face, 'kps') and face.kps is not None:
                for point in face.kps:
                    x, y = point.astype(int)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        return frame

# Singleton recognizer
recognizer = VideoFaceRecognizer()

GMAIL_USER = 'tutifytest@gmail.com'
GMAIL_PASS = 'bkpiqehtbfbpkajx'
DOWNLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'downloads'))
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/video_status")
def video_status():
    frame = camera.get_frame()
    if frame is None or np.count_nonzero(frame) == 0:
        return jsonify({"status": "no_face_detected"})
    results = recognizer.detect_and_recognize_faces(frame)
    if not results:
        return jsonify({"status": "no_face_detected"})
    for result in results:
        if result['name'] == "Spoof" or not result.get('is_live', True):
            return jsonify({"status": "spoof_detected"})
        elif result['name'] != "Unknown":
            return jsonify({"status": "face_detected"})
    return jsonify({"status": "no_face_detected"})

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'summary': ['no_face_detected'], 'error': 'No video uploaded'}), 400
    video_file = request.files['video']
    temp_path = None
    try:
        # Read the uploaded file into memory once
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            temp.write(video_bytes)
            temp_path = temp.name
        # First pass: check for spoof
        cap = cv2.VideoCapture(temp_path)
        total_faces = 0
        spoof_count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            results = recognizer.detect_and_recognize_faces(frame)
            if not results:
                frame_idx += 1
                continue
            for result in results:
                face = result['face']
                if not hasattr(face, 'embedding'):
                    continue
                print(f"Frame {frame_idx}: is_live={result['is_live']}, name={result['name']}, live_score={result.get('live_score')}")
                total_faces += 1
                is_live = result['is_live'] and result['name'] != 'Spoof'
                if not is_live:
                    spoof_count += 1
            frame_idx += 1
        cap.release()
        if total_faces > 0 and spoof_count / total_faces > 0.3:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'summary': ['spoof_detected']})
        # Second pass: check for live faces
        cap = cv2.VideoCapture(temp_path)
        tracked_faces = []  # Each: {'embedding': np.ndarray, 'is_live': bool}
        SIM_THRESHOLD = 0.75
        live_detected = False
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            results = recognizer.detect_and_recognize_faces(frame)
            if not results:
                continue
            for result in results:
                face = result['face']
                if not hasattr(face, 'embedding'):
                    continue
                embedding = face.embedding
                is_live = result['is_live'] and result['name'] != 'Spoof'
                if is_live:
                    matched = False
                    for tracked in tracked_faces:
                        sim = float(np.dot(embedding, tracked['embedding']) / (np.linalg.norm(embedding) * np.linalg.norm(tracked['embedding'])))
                        if sim > SIM_THRESHOLD:
                            matched = True
                            break
                    if not matched:
                        tracked_faces.append({'embedding': embedding, 'is_live': True})
                        live_detected = True
        cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        summary = []
        if live_detected:
            summary.append('live_detected')
        if not summary:
            summary = ['no_face_detected']
        return jsonify({'summary': summary})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'summary': ['error'], 'error': str(e)}), 500

@app.route('/send_album', methods=['POST'])
def send_album():
    email = request.form.get('email')
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    video_file = request.files['video']
    temp_path = None
    try:
        # Read video into memory
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            temp.write(video_bytes)
            temp_path = temp.name
        # First pass: check for spoof
        cap = cv2.VideoCapture(temp_path)
        total_faces = 0
        spoof_count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            results = recognizer.detect_and_recognize_faces(frame)
            if not results:
                frame_idx += 1
                continue
            for result in results:
                face = result['face']
                if not hasattr(face, 'embedding'):
                    continue
                print(f"Frame {frame_idx}: is_live={result['is_live']}, name={result['name']}, live_score={result.get('live_score')}")
                total_faces += 1
                is_live = result['is_live'] and result['name'] != 'Spoof'
                if not is_live:
                    spoof_count += 1
            frame_idx += 1
        cap.release()
        if total_faces > 0 and spoof_count / total_faces > 0.3:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'summary': ['spoof_detected']})
        # Respond immediately to UI that processing has started
        def process_album_and_send_email(video_bytes, email):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
                    temp.write(video_bytes)
                    temp_path = temp.name
                cap = cv2.VideoCapture(temp_path)
                matched_names = set()
                SIM_THRESHOLD = 0.4
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    results = recognizer.detect_and_recognize_faces(frame)
                    if not results:
                        continue
                    for result in results:
                        if hasattr(result['face'], 'embedding'):
                            for name, face_data in recognizer.known_faces.items():
                                similarity = float(np.dot(result['face'].embedding, face_data['embedding']) / (
                                    np.linalg.norm(result['face'].embedding) * np.linalg.norm(face_data['embedding'])
                                ))
                                if similarity >= SIM_THRESHOLD:
                                    matched_names.add(name)
                cap.release()
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                if not matched_names:
                    # Optionally, you could email the user that no matches were found
                    return
                # Zip matched images
                zip_token = str(uuid.uuid4())
                zip_filename = f"album_{zip_token}.zip"
                zip_path = os.path.join(DOWNLOAD_DIR, zip_filename)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for name in matched_names:
                        img_path = recognizer.known_faces[name]['image_path']
                        zipf.write(img_path, arcname=os.path.basename(img_path))
                # Generate download link
                download_url = f"http://localhost:5000/download/{zip_filename}"
                # Send email
                msg = MIMEMultipart()
                msg['From'] = formataddr(("FaceRec App", GMAIL_USER))
                msg['To'] = email
                msg['Subject'] = "Your FaceMatch Album Download Link"
                body = f"Hello,\n\nYour personalized album is ready! Download it here: {download_url}\n\nBest regards,\nFaceRec Team"
                msg.attach(MIMEText(body, 'plain'))
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login(GMAIL_USER, GMAIL_PASS)
                    server.send_message(msg)
            except Exception as e:
                print(f"Error in background album/email processing: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
        threading.Thread(target=process_album_and_send_email, args=(video_bytes, email), daemon=True).start()
        return jsonify({'processing': True})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<zipname>')
def download_zip(zipname):
    print(f"DOWNLOAD_DIR: {DOWNLOAD_DIR}")
    print(f"Requested zip: {zipname}")
    print(f"Full path: {os.path.join(DOWNLOAD_DIR, zipname)}")
    print(f"Exists: {os.path.exists(os.path.join(DOWNLOAD_DIR, zipname))}")
    if not os.path.exists(os.path.join(DOWNLOAD_DIR, zipname)):
        return "File not found", 404
    return send_from_directory(DOWNLOAD_DIR, zipname, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False) 