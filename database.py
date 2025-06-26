import os
import cv2
import numpy as np
import json
import time
import logging
from typing import List, Dict, Any
from config import FaceRecognitionConfig

logger = logging.getLogger(__name__)

class FaceDatabase:
    """Manages the database of known faces"""
    
    def __init__(self, config: FaceRecognitionConfig):
        self.config = config
        self.embeddings: List[np.ndarray] = []
        self.names: List[str] = []
        self.face_images: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def load_from_directory(self) -> int:
        """Load faces from the images directory"""
        logger.info(f"Loading faces from {self.config.images_dir}")
        
        if not os.path.exists(self.config.images_dir):
            logger.error(f"Images directory {self.config.images_dir} does not exist")
            return 0
            
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        loaded_count = 0
        
        for filename in os.listdir(self.config.images_dir):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(self.config.images_dir, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    # Load image
                    image = cv2.imread(filepath)
                    if image is None:
                        logger.warning(f"Could not load image: {filename}")
                        continue
                        
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Store face image
                    self.face_images.append(image_rgb)
                    self.names.append(name)
                    
                    # Add metadata
                    metadata = {
                        'filename': filename,
                        'filepath': filepath,
                        'image_size': image.shape,
                        'added_date': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    self.metadata.append(metadata)
                    
                    loaded_count += 1
                    logger.info(f"Loaded face: {name} ({image.shape[1]}x{image.shape[0]})")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    
        logger.info(f"Successfully loaded {loaded_count} faces")
        return loaded_count
    
    def add_face(self, name: str, embedding: np.ndarray, face_image: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a new face to the database"""
        self.names.append(name)
        self.embeddings.append(embedding)
        self.face_images.append(face_image)
        
        if metadata is None:
            metadata = {}
        metadata['added_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.metadata.append(metadata)
        
        logger.info(f"Added new face: {name}")
    
    def save_database(self):
        """Save database to JSON file"""
        try:
            data = {
                'names': self.names,
                'embeddings': [emb.tolist() for emb in self.embeddings],
                'metadata': self.metadata
            }
            
            with open(self.config.database_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Database saved to {self.config.database_path}")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def load_database(self) -> bool:
        """Load database from JSON file"""
        if not os.path.exists(self.config.database_path):
            logger.info(f"Database file {self.config.database_path} not found")
            return False
            
        try:
            with open(self.config.database_path, 'r') as f:
                data = json.load(f)
            
            self.names = data['names']
            self.embeddings = [np.array(emb) for emb in data['embeddings']]
            self.metadata = data['metadata']
            
            logger.info(f"Loaded database with {len(self.names)} faces")
            return True
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False
    
    def get_count(self) -> int:
        """Get number of faces in database"""
        return len(self.names)
    
    def get_names(self) -> List[str]:
        """Get list of known names"""
        return self.names.copy() 