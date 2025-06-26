#!/usr/bin/env python3
"""
Face Detection and Recognition Video Script using insightface library
This script detects and recognizes faces in video files or webcam feed using a database of known faces from the 'images' folder.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import argparse
import os
import time
import pickle


class FaceRecognizer:
    def __init__(self, model_name='buffalo_l'):
        """
        Initialize the face recognizer with insightface.
        
        Args:
            model_name (str): Name of the insightface model to use
        """
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"Face recognition model '{model_name}' loaded successfully!")
        
        # Database of known faces
        self.known_faces = {}
        self.known_face_embeddings = []
        self.known_face_names = []
        
        # Load known faces from images folder
        self.load_known_faces()
    
    def load_known_faces(self, images_folder="images"):
        """
        Load known faces from the images folder.
        
        Args:
            images_folder (str): Path to folder containing reference images
        """
        if not os.path.exists(images_folder):
            print(f"Warning: Images folder '{images_folder}' not found. No known faces loaded.")
            return
        
        print(f"Loading known faces from '{images_folder}' folder...")
        
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
                        print(f"  Loaded: {person_name} from {filename}")
                    else:
                        print(f"  No face found in: {filename}")
                        
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
        
        print(f"Loaded {loaded_count} known faces from database.")
        print("-" * 50)
    
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


def process_video_file(video_path, output_path=None, show_preview=True):
    """
    Process a video file for face detection and recognition.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output video (optional)
        show_preview (bool): Whether to show the video preview
    """
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print("-" * 50)
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
    frame_count = 0
    total_faces_detected = 0
    recognition_stats = {}
    start_time = time.time()
    
    print("Processing video... Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and recognize faces
        recognition_results = recognizer.detect_and_recognize_faces(frame)
        
        # Draw faces on frame
        annotated_frame = recognizer.draw_faces(frame, recognition_results)
        
        # Update statistics
        faces_count = len(recognition_results)
        total_faces_detected += faces_count
        
        for result in recognition_results:
            name = result['name']
            if name not in recognition_stats:
                recognition_stats[name] = 0
            recognition_stats[name] += 1
        
        # Add info text
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Faces: {faces_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show progress
        if frame_count % 30 == 0:  # Update every 30 frames
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | Faces: {faces_count}")
        
        # Write frame to output video
        if writer:
            writer.write(annotated_frame)
        
        # Show preview
        if show_preview:
            cv2.imshow('Face Recognition', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"frame_{frame_count}_faces_{faces_count}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved as: {save_path}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print("-" * 50)
    print("Processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    # Print recognition statistics
    if recognition_stats:
        print("\nRecognition Statistics:")
        for name, count in recognition_stats.items():
            print(f"  {name}: {count} detections")


def process_webcam(show_preview=True):
    """
    Process webcam feed for real-time face detection and recognition.
    
    Args:
        show_preview (bool): Whether to show the video preview
    """
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam face recognition started!")
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Mirror the frame horizontally (like a mirror)
        frame = cv2.flip(frame, 1)
        
        # Detect and recognize faces
        recognition_results = recognizer.detect_and_recognize_faces(frame)
        
        # Draw faces on frame
        annotated_frame = recognizer.draw_faces(frame, recognition_results)
        
        # Add info text
        faces_count = len(recognition_results)
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(annotated_frame, f"Faces: {faces_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show preview
        if show_preview:
            cv2.imshow('Webcam Face Recognition', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"webcam_frame_{frame_count}_faces_{faces_count}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved as: {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("-" * 50)
    print("Webcam processing stopped!")


def main():
    """Main function to handle command line arguments and run face recognition."""
    parser = argparse.ArgumentParser(
        description="Face Detection and Recognition in Video using insightface library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection_video.py --webcam                    # Use webcam
  python face_detection_video.py -v video.mp4                # Process video file
  python face_detection_video.py -v video.mp4 -o output.mp4  # Save output video
  python face_detection_video.py -v video.mp4 --no-preview   # Process without preview
        """
    )
    
    parser.add_argument(
        '-v', '--video',
        help='Path to input video file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to save output video file'
    )
    
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam instead of video file'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Don\'t show video preview (faster processing)'
    )
    
    parser.add_argument(
        '--model',
        default='buffalo_l',
        help='insightface model name (default: buffalo_l)'
    )
    
    parser.add_argument(
        '--images-folder',
        default='images',
        help='Folder containing reference face images (default: images)'
    )
    
    args = parser.parse_args()
    
    print("Face Detection and Recognition Video Script")
    print("=" * 50)
    
    # Check if insightface is available
    try:
        import insightface
        print("insightface library loaded successfully!")
    except ImportError:
        print("Error: insightface library not found!")
        print("Please install it using: pip install insightface")
        return
    
    # Process based on arguments
    if args.webcam:
        process_webcam(not args.no_preview)
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file '{args.video}' not found!")
            return
        process_video_file(args.video, args.output, not args.no_preview)
    else:
        print("Error: Please specify either --webcam or --video option")
        parser.print_help()


if __name__ == "__main__":
    main() 