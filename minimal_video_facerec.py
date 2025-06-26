import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
from typing import List, Dict, Tuple, Optional
import time
import argparse
from collections import deque

class MinimalVideoFaceRecognition:
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        Initialize the video face recognition system with InsightFace
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
        """
        print("Initializing InsightFace...")
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Store known faces: {name: embedding}
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_database_file = 'minimal_face_db.json'
        self.similarity_threshold = 0.4
        
        # Video processing settings
        self.frame_skip = 2  # Process every nth frame
        self.display_fps = True
        self.show_confidence = True
        
        # Anti-flickering settings
        self.result_buffer = deque(maxlen=3)  # Buffer for smoothing results
        self.last_results = []
        self.frame_count = 0
        
        # Load existing database if available
        self.load_database()
    
    def load_database(self):
        """Load face database from JSON file"""
        if os.path.exists(self.face_database_file):
            try:
                with open(self.face_database_file, 'r') as f:
                    data = json.load(f)
                    for name, embedding_list in data.items():
                        self.known_faces[name] = np.array(embedding_list)
                print(f"Loaded {len(self.known_faces)} known faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save_database(self):
        """Save face database to JSON file"""
        try:
            data = {name: embedding.tolist() for name, embedding in self.known_faces.items()}
            with open(self.face_database_file, 'w') as f:
                json.dump(data, f)
            print(f"Saved {len(self.known_faces)} faces to database")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Face embedding vector or None if no face detected
        """
        faces = self.app.get(image)
        if faces:
            return faces[0].embedding
        return None
    
    def add_face_from_image(self, name: str, image_path: str) -> bool:
        """
        Add a new face to the database from image file
        
        Args:
            name: Person's name
            image_path: Path to image file
            
        Returns:
            True if face was added successfully
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False
        
        embedding = self.extract_face_embedding(image)
        if embedding is not None:
            self.known_faces[name] = embedding
            self.save_database()
            print(f"Added face for {name}")
            return True
        else:
            print("No face detected in image")
            return False
    
    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in video frame
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of (name, similarity_score, bbox) tuples
        """
        faces = self.app.get(frame)
        results = []
        
        for face in faces:
            best_match = None
            best_score = 0
            
            for name, known_embedding in self.known_faces.items():
                similarity = np.dot(face.embedding, known_embedding) / (
                    np.linalg.norm(face.embedding) * np.linalg.norm(known_embedding)
                )
                
                if similarity > best_score and similarity > self.similarity_threshold:
                    best_score = similarity
                    best_match = name
            
            bbox = tuple(face.bbox.astype(int))
            if best_match:
                results.append((best_match, best_score, bbox))
            else:
                results.append(("Unknown", 0.0, bbox))
        
        return results
    
    def smooth_results(self, new_results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Smooth recognition results to reduce flickering
        
        Args:
            new_results: Current frame results
            
        Returns:
            Smoothed results
        """
        if not new_results:
            # If no faces detected, gradually fade out previous results
            if self.last_results:
                # Keep last results for a few frames to prevent sudden disappearance
                return self.last_results
            return []
        
        # Add new results to buffer
        self.result_buffer.append(new_results)
        
        # If buffer is not full, use current results
        if len(self.result_buffer) < 2:
            self.last_results = new_results
            return new_results
        
        # Smooth results by averaging bounding boxes and keeping consistent names
        smoothed_results = []
        
        for i, current_face in enumerate(new_results):
            current_name, current_score, current_bbox = current_face
            
            # Find matching face in previous results based on position
            best_match = None
            best_distance = float('inf')
            
            for prev_name, prev_score, prev_bbox in self.last_results:
                # Calculate distance between bounding box centers
                curr_center = ((current_bbox[0] + current_bbox[2]) // 2, (current_bbox[1] + current_bbox[3]) // 2)
                prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                
                if distance < best_distance and distance < 100:  # Threshold for matching
                    best_distance = distance
                    best_match = (prev_name, prev_score, prev_bbox)
            
            if best_match:
                # Smooth bounding box
                prev_name, prev_score, prev_bbox = best_match
                smoothed_bbox = tuple(int(0.7 * prev + 0.3 * curr) for prev, curr in zip(prev_bbox, current_bbox))
                
                # Use previous name if it's the same person (high confidence)
                if current_score > 0.8 and prev_name != "Unknown":
                    final_name = prev_name
                    final_score = max(current_score, prev_score)
                else:
                    final_name = current_name
                    final_score = current_score
                
                smoothed_results.append((final_name, final_score, smoothed_bbox))
            else:
                # New face detected
                smoothed_results.append(current_face)
        
        self.last_results = smoothed_results
        return smoothed_results
    
    def draw_recognition_results(self, frame: np.ndarray, results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw recognition results on video frame
        
        Args:
            frame: Input frame
            results: Recognition results
            
        Returns:
            Frame with annotations
        """
        result_frame = frame.copy()
        
        for name, score, (x1, y1, x2, y2) in results:
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw name and confidence
            if self.show_confidence:
                label = f"{name}: {score:.2f}"
            else:
                label = name
            
            # Calculate text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
            
            # Draw text background
            cv2.rectangle(result_frame, (text_x, text_y - text_size[1]), 
                         (text_x + text_size[0], text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(result_frame, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def process_video(self, video_source: str = '0'):
        """
        Process video stream for face recognition
        
        Args:
            video_source: Video source (0 for webcam, or video file path)
        """
        try:
            # Open video capture
            if video_source.isdigit():
                cap = cv2.VideoCapture(int(video_source))
                print(f"Opening webcam {video_source}")
            else:
                cap = cv2.VideoCapture(video_source)
                print(f"Opening video file: {video_source}")
            
            if not cap.isOpened():
                print("Error: Could not open video source")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video: {width}x{height} @ {fps:.1f} FPS")
            print("Press 'q' to quit, 's' to save current frame")
            
            frame_count = 0
            start_time = time.time()
            last_fps_time = start_time
            fps_counter = 0
            current_fps = 0.0  # Initialize FPS variable
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                frame_count += 1
                fps_counter += 1
                
                # Process every nth frame for performance
                if frame_count % self.frame_skip == 0:
                    # Recognize faces
                    results = self.recognize_faces_in_frame(frame)
                    
                    # Smooth results to reduce flickering
                    smoothed_results = self.smooth_results(results)
                    
                    # Draw results
                    frame = self.draw_recognition_results(frame, smoothed_results)
                
                # Update FPS display every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    current_fps = fps_counter / (current_time - last_fps_time)
                    last_fps_time = current_time
                    fps_counter = 0
                
                # Display FPS
                if self.display_fps:
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display face count
                if self.last_results:
                    cv2.putText(frame, f"Faces: {len(self.last_results)}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Minimal Face Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('h'):
                    # Toggle help
                    print("Controls: q=quit, s=save frame, h=help")
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing video: {e}")
    
    def list_known_faces(self):
        """List all known faces in database"""
        if self.known_faces:
            print("Known faces:")
            for name in self.known_faces.keys():
                print(f"  - {name}")
        else:
            print("No faces in database")

def main():
    parser = argparse.ArgumentParser(description='Minimal Video Face Recognition')
    parser.add_argument('--source', '-s', default='0', 
                       help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--add', '-a', nargs=2, metavar=('NAME', 'IMAGE_PATH'),
                       help='Add face to database: NAME IMAGE_PATH')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List known faces')
    parser.add_argument('--threshold', '-t', type=float, default=0.4,
                       help='Similarity threshold (default: 0.4)')
    parser.add_argument('--skip', type=int, default=2,
                       help='Process every nth frame (default: 2)')
    
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_rec = MinimalVideoFaceRecognition()
    face_rec.similarity_threshold = args.threshold
    face_rec.frame_skip = args.skip
    
    if args.add:
        name, image_path = args.add
        face_rec.add_face_from_image(name, image_path)
    elif args.list:
        face_rec.list_known_faces()
    else:
        print("=== Minimal Video Face Recognition ===")
        print(f"Similarity threshold: {face_rec.similarity_threshold}")
        print(f"Frame skip: {face_rec.frame_skip}")
        print("Controls: q=quit, s=save frame, h=help")
        print()
        
        # Start video processing
        face_rec.process_video(args.source)

if __name__ == "__main__":
    main() 