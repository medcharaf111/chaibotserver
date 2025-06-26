import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import logging
import os
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Simple and effective face recognition system using insightface"""
    
    def __init__(self, model_name='buffalo_l', images_folder="images"):
        """
        Initialize the face recognition system.
        
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
    
    def recognize_face(self, face_embedding, threshold=0.6):
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
        Detect and recognize faces in a frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            list: List of detected faces with recognition results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(rgb_frame)
        
        # Recognize each face
        recognition_results = []
        for face in faces:
            if hasattr(face, 'embedding'):
                person_name, confidence = self.recognize_face(face.embedding)
                recognition_results.append({
                    'face': face,
                    'name': person_name,
                    'confidence': confidence
                })
            else:
                recognition_results.append({
                    'face': face,
                    'name': 'Unknown',
                    'confidence': 0
                })
        
        return recognition_results
    
    def draw_faces(self, frame, recognition_results):
        """
        Draw bounding boxes and recognition results around detected faces.
        
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
            
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Choose color based on recognition result
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add name and confidence label
            label = f"{name} ({confidence:.2f})"
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
    
    def get_database_count(self):
        """Get number of faces in database"""
        return len(self.known_face_names)
    
    def get_known_names(self):
        """Get list of known names"""
        return self.known_face_names.copy() 