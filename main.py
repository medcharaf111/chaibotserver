#!/usr/bin/env python3
"""
Face Detection and Recognition System using insightface library
This script provides real-time face detection and recognition with a simple, effective approach.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import argparse
import os
import time
import logging
import json
# Add DeepFace Fasnet import for liveness detection
from deepface.models.spoofing.FasNet import Fasnet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleFaceRecognizer:
    """Simple and effective face recognition system using insightface"""
    
    def __init__(self, model_name='buffalo_s', images_folder="images"):
        """
        Initialize the face recognizer with insightface.
        
        Args:
            model_name (str): Name of the insightface model to use
            images_folder (str): Path to folder containing reference images
        """
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info(f"Face recognition model '{model_name}' loaded successfully!")
        
        # Database of known faces
        self.known_faces = {}
        self.known_face_embeddings = []
        self.known_face_names = []
        
        # Store images folder path
        self.images_folder = images_folder
        
        # Load known faces from images folder
        self.load_known_faces(images_folder)
        # Initialize liveness detector
        self.liveness_detector = Fasnet()
    
    def load_known_faces(self, images_folder="images"):
        """
        Load known faces from the images folder.
        
        Args:
            images_folder (str): Path to folder containing reference images
        """
        if not os.path.exists(images_folder):
            logger.warning(f"Images folder '{images_folder}' not found. No known faces loaded.")
            return
        
        logger.info(f"Loading known faces from '{images_folder}' folder...")
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        loaded_count = 0
        for filename in os.listdir(images_folder):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_path = os.path.join(images_folder, filename)
                
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the image
                    faces = self.app.get(rgb_image)
                    
                    if len(faces) > 0:
                        # Use the first face found
                        face = faces[0]
                        
                        # Get face embedding
                        embedding = face.embedding
                        
                        # Use filename (without extension) as person name
                        person_name = os.path.splitext(filename)[0]
                        
                        # Store face data
                        self.known_faces[person_name] = {
                            'embedding': embedding,
                            'image_path': image_path
                        }
                        self.known_face_embeddings.append(embedding)
                        self.known_face_names.append(person_name)
                        
                        loaded_count += 1
                        logger.info(f"  Loaded: {person_name} from {filename}")
                    else:
                        logger.warning(f"  No face found in: {filename}")
                        
                except Exception as e:
                    logger.error(f"  Error loading {filename}: {e}")
        
        logger.info(f"Loaded {loaded_count} known faces from database.")
        logger.info("-" * 50)
    
    def reload_database(self):
        """
        Reload the database from the images folder.
        This is useful after adding new faces to ensure they are immediately available.
        """
        # Clear current database
        self.known_faces = {}
        self.known_face_embeddings = []
        self.known_face_names = []
        
        # Reload from images folder
        self.load_known_faces(self.images_folder)
        logger.info(f"Database reloaded. Now contains {len(self.known_face_names)} faces")
    
    def add_new_face(self, name, face_embedding, face_image, full_frame=None):
        """
        Add a new face to the database and save the image.
        
        Args:
            name (str): Name of the person
            face_embedding: Face embedding
            face_image: Face image (BGR format) - can be face region or full frame
            full_frame: Full frame image (optional, for better quality)
        """
        try:
            # Create images folder if it doesn't exist
            if not os.path.exists(self.images_folder):
                os.makedirs(self.images_folder)
            
            # Save face image
            image_filename = f"{name}.jpg"
            image_path = os.path.join(self.images_folder, image_filename)
            
            # Ensure filename is unique
            counter = 1
            while os.path.exists(image_path):
                image_filename = f"{name}_{counter}.jpg"
                image_path = os.path.join(self.images_folder, image_filename)
                counter += 1
            
            # Use full frame if available, otherwise use face region
            image_to_save = full_frame if full_frame is not None else face_image
            
            # Ensure minimum size for face detection
            if image_to_save.shape[0] < 100 or image_to_save.shape[1] < 100:
                logger.warning(f"Image too small ({image_to_save.shape}), using face region instead")
                image_to_save = face_image
            
            # Save the image with high quality
            cv2.imwrite(image_path, image_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Verify the image was saved successfully
            if not os.path.exists(image_path):
                logger.error(f"Failed to save image to {image_path}")
                return False
            
            # Verify the saved image can be loaded and contains a face
            saved_image = cv2.imread(image_path)
            if saved_image is None:
                logger.error(f"Failed to load saved image from {image_path}")
                return False
            
            # Convert to RGB and check if face is detectable
            saved_rgb = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
            faces = self.app.get(saved_rgb)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in saved image {image_path}")
                logger.warning(f"Image size: {saved_image.shape}")
                # Still add to database but with warning
                logger.warning("Adding face to database despite detection failure")
            
            # Add to database
            self.known_faces[name] = {
                'embedding': face_embedding,
                'image_path': image_path
            }
            self.known_face_embeddings.append(face_embedding)
            self.known_face_names.append(name)
            
            logger.info(f"Successfully added new face: {name} (saved as {image_filename})")
            logger.info(f"Image saved to: {image_path}")
            logger.info(f"Image size: {saved_image.shape}")
            logger.info(f"Database now contains {len(self.known_face_names)} faces")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding new face: {e}")
            return False
    
    def recognize_face(self, face_embedding, threshold=0.4):
        """
        Recognize a face by comparing its embedding with known faces.
        
        Args:
            face_embedding: Face embedding to recognize
            threshold (float): Similarity threshold for recognition
            
        Returns:
            tuple: (person_name, confidence_score) or (None, 0) if not recognized
        """
        if len(self.known_face_embeddings) == 0:
            return None, 0
        
        # Calculate similarities with all known faces
        similarities = []
        for known_embedding in self.known_face_embeddings:
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, known_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
            )
            similarities.append(similarity)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= threshold:
            return self.known_face_names[best_match_idx], best_similarity
        else:
            return "Unknown", best_similarity
    
    def detect_and_recognize_faces(self, frame):
        """
        Detect and recognize faces in a frame, with liveness detection.
        Args:
            frame: BGR image frame from OpenCV
        Returns:
            list: List of detected faces with recognition and liveness results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        faces = self.app.get(rgb_frame)
        # Recognize each face with liveness detection
        recognition_results = []
        for face in faces:
            # Get bounding box as x, y, w, h
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            # Run liveness detection
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
        """
        Draw bounding boxes and recognition/liveness results around detected faces.
        Args:
            frame: BGR image frame
            recognition_results: List of recognition results
        Returns:
            frame: Annotated frame
        """
        for i, result in enumerate(recognition_results):
            face = result['face']
            name = result['name']
            confidence = result['confidence']
            is_live = result.get('is_live', None)
            live_score = result.get('live_score', None)
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # Choose color based on recognition/liveness result
            if name == "Spoof" or not is_live:
                color = (0, 165, 255)  # Orange for spoof
            elif name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Add name, confidence, and liveness label
            if is_live is not None:
                liveness_label = f"Live: {is_live} ({live_score:.2f})"
            else:
                liveness_label = "Live: N/A"
            label = f"{name} ({confidence:.2f}) | {liveness_label}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Draw facial landmarks (optional)
            if hasattr(face, 'kps') and face.kps is not None:
                for point in face.kps:
                    x, y = point.astype(int)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        return frame

def process_webcam(show_preview=True, images_folder="images"):
    """
    Process webcam feed for real-time face detection and recognition.
    """
    recognizer = SimpleFaceRecognizer(images_folder=images_folder)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info("Webcam face recognition started!")
    logger.info("Press 'q' to quit, 's' to save current frame, 'a' to add new face")
    logger.info("-" * 50)
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.flip(frame, 1)
            recognition_results = recognizer.detect_and_recognize_faces(frame)
            annotated_frame = recognizer.draw_faces(frame, recognition_results)
            faces_count = len(recognition_results)
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(annotated_frame, f"Faces: {faces_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            db_count = len(recognizer.known_face_names)
            cv2.putText(annotated_frame, f"Database: {db_count} faces", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save frame, 'a' to add face", (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if show_preview:
                cv2.imshow('Face Recognition System', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_path = f"webcam_frame_{frame_count}_faces_{faces_count}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    logger.info(f"Frame saved to {save_path}")
                elif key == ord('a'):
                    logger.info("Add new face feature not implemented in this mode.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to handle command line arguments and run face recognition."""
    parser = argparse.ArgumentParser(
        description="Face Detection and Recognition System using insightface library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use webcam with default settings
  python main.py --no-preview                       # Process without preview
  python main.py --images-folder images_clean       # Use different images folder
  python main.py --model buffalo_s                  # Use different model

Controls:
  Press 'q' to quit
  Press 's' to save current frame
  Press 'a' to add new face to database
        """
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Don\'t show video preview (faster processing)'
    )
    
    parser.add_argument(
        '--model',
        default='buffalo_s',
        help='insightface model name (default: buffalo_s)'
    )
    
    parser.add_argument(
        '--images-folder',
        default='images',
        help='Folder containing reference face images (default: images)'
    )
    
    args = parser.parse_args()
    
    logger.info("Face Detection and Recognition System")
    logger.info("=" * 50)
    
    # Check if insightface is available
    try:
        import insightface
        logger.info("insightface library loaded successfully!")
    except ImportError:
        logger.error("insightface library not found!")
        logger.error("Please install it using: pip install insightface")
        return
    
    # Process webcam
    process_webcam(not args.no_preview, args.images_folder)

if __name__ == "__main__":
    main() 