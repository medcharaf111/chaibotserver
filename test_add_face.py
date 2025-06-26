#!/usr/bin/env python3
"""
Test script to verify add face functionality
"""

import cv2
import os
import time
from recognition import FaceRecognitionSystem

def test_add_face():
    """Test the add face functionality"""
    
    # Initialize recognition system
    recognizer = FaceRecognitionSystem(images_folder="test_images")
    
    # Create test images folder
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    
    print("Testing add face functionality...")
    print(f"Initial database count: {recognizer.get_database_count()}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Press 'a' to add a face, 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            results = recognizer.detect_and_recognize_faces(frame)
            
            # Draw results
            annotated_frame = recognizer.draw_faces(frame, results)
            
            # Add info
            cv2.putText(annotated_frame, f"Database: {recognizer.get_database_count()} faces", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Press 'a' to add face, 'q' to quit", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Test Add Face', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if results:
                    face_result = results[0]
                    face = face_result['face']
                    
                    # Get face region with padding
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Add padding
                    height, width = frame.shape[:2]
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)
                    
                    face_region = frame[y1:y2, x1:x2]
                    
                    if face_region.size > 0:
                        # Test name
                        test_name = f"test_person_{int(time.time())}"
                        
                        print(f"\nAdding test face: {test_name}")
                        success = recognizer.add_new_face(test_name, face.embedding, face_region)
                        
                        if success:
                            print(f"✅ Successfully added {test_name}")
                            print(f"📁 Database count: {recognizer.get_database_count()}")
                            
                            # Verify file exists
                            expected_path = os.path.join("test_images", f"{test_name}.jpg")
                            if os.path.exists(expected_path):
                                print(f"✅ Image file exists: {expected_path}")
                                file_size = os.path.getsize(expected_path)
                                print(f"📏 File size: {file_size} bytes")
                            else:
                                print(f"❌ Image file not found: {expected_path}")
                        else:
                            print("❌ Failed to add face")
                    else:
                        print("❌ Invalid face region")
                else:
                    print("❌ No face detected")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # List all files in test_images
        print(f"\nFiles in test_images folder:")
        if os.path.exists("test_images"):
            for filename in os.listdir("test_images"):
                filepath = os.path.join("test_images", filename)
                file_size = os.path.getsize(filepath)
                print(f"  {filename} ({file_size} bytes)")
        else:
            print("  test_images folder not found")

if __name__ == "__main__":
    test_add_face() 