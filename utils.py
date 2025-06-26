import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'CUDAExecutionProvider' in providers
    except ImportError:
        logger.warning("ONNX Runtime not available")
        return False

def get_gpu_info() -> dict:
    """Get GPU information"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_info = {
            'cuda_available': 'CUDAExecutionProvider' in providers,
            'providers': providers
        }
        return gpu_info
    except ImportError:
        return {'cuda_available': False, 'providers': []}

def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)
    
    return cv2.resize(image, (new_width, new_height))

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Apply basic image enhancement"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def validate_image_path(image_path: str) -> bool:
    """Validate if image path exists and is readable"""
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {image_path}: {e}")
        return False

def calculate_face_quality(face_image: np.ndarray) -> float:
    """Calculate face image quality score (0-1)"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (typical values are 0-1000)
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # Calculate brightness score
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Calculate contrast score
        contrast = np.std(gray)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # Combine scores
        quality_score = (sharpness_score + brightness_score + contrast_score) / 3.0
        
        return quality_score
        
    except Exception as e:
        logger.error(f"Error calculating face quality: {e}")
        return 0.0

def draw_face_landmarks(image: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw face landmarks on image"""
    if landmarks is None:
        return image
    
    for point in landmarks:
        cv2.circle(image, tuple(point.astype(int)), 2, color, -1)
    
    return image

def create_face_mosaic(face_images: List[np.ndarray], max_width: int = 800) -> np.ndarray:
    """Create a mosaic of face images"""
    if not face_images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize all images to same size
    target_size = (100, 100)
    resized_images = []
    
    for img in face_images:
        resized = cv2.resize(img, target_size)
        resized_images.append(resized)
    
    # Calculate grid dimensions
    n_images = len(resized_images)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    # Create mosaic
    mosaic = np.zeros((rows * target_size[0], cols * target_size[1], 3), dtype=np.uint8)
    
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        y1 = row * target_size[0]
        y2 = (row + 1) * target_size[0]
        x1 = col * target_size[1]
        x2 = (col + 1) * target_size[1]
        mosaic[y1:y2, x1:x2] = img
    
    # Resize to max width if needed
    if mosaic.shape[1] > max_width:
        scale = max_width / mosaic.shape[1]
        new_height = int(mosaic.shape[0] * scale)
        mosaic = cv2.resize(mosaic, (max_width, new_height))
    
    return mosaic 