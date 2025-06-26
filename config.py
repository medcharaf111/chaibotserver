from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FaceRecognitionConfig:
    """Configuration for the face recognition system"""
    # GPU settings
    use_gpu: bool = True
    gpu_id: int = 0
    
    # Recognition settings
    recognition_threshold: float = 0.6
    min_face_size: int = 80
    max_faces: int = 10
    
    # Performance settings
    processing_interval: float = 0.05  # seconds
    batch_size: int = 4
    
    # Display settings
    show_confidence: bool = True
    show_fps: bool = True
    show_landmarks: bool = False
    
    # Database settings
    database_path: str = "face_database.json"
    images_dir: str = "images"
    
    # Anti-spoofing settings
    enable_anti_spoofing: bool = False
    anti_spoofing_threshold: float = 0.5 