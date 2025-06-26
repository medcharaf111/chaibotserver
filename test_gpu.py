#!/usr/bin/env python3
"""
Test script for system functionality (GPU code removed)
"""

import sys
import logging
from utils import check_gpu_availability, get_gpu_info
from config import FaceRecognitionConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_support():
    """Test GPU support"""
    print("=== GPU Support Test ===")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    print(f"CUDA GPU Available: {gpu_available}")
    
    # Get detailed GPU info
    gpu_info = get_gpu_info()
    print(f"GPU Info: {gpu_info}")
    
    return gpu_available

def test_imports():
    """Test all module imports"""
    print("\n=== Module Import Test ===")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import insightface
        print(f"✓ InsightFace imported successfully")
    except ImportError as e:
        print(f"✗ InsightFace import failed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"✗ ONNX Runtime import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration system"""
    print("\n=== Configuration Test ===")
    
    try:
        config = FaceRecognitionConfig()
        print(f"✓ Configuration created successfully")
        print(f"  GPU enabled: {config.use_gpu}")
        print(f"  Recognition threshold: {config.recognition_threshold}")
        print(f"  Processing interval: {config.processing_interval}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n=== Database Test ===")
    
    try:
        from database import FaceDatabase
        config = FaceRecognitionConfig()
        db = FaceDatabase(config)
        print(f"✓ Database created successfully")
        print(f"  Database count: {db.get_count()}")
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Face Recognition System - GPU Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test GPU support
    gpu_ok = test_gpu_support()
    
    # Test configuration
    config_ok = test_config()
    
    # Test database
    db_ok = test_database()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"GPU Support: {'✓' if gpu_ok else '✗'}")
    print(f"Configuration: {'✓' if config_ok else '✗'}")
    print(f"Database: {'✓' if db_ok else '✗'}")
    
    if all([imports_ok, config_ok, db_ok]):
        print("\n✓ All tests passed! System is ready to run.")
        if gpu_ok:
            print("✓ GPU acceleration is available!")
        else:
            print("⚠ GPU acceleration not available, will use CPU")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 