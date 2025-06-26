#!/usr/bin/env python3
"""
Face Matcher - Image-based Face Recognition System
Takes an input image and finds matching faces in the images folder with accuracy scores.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import argparse
import json
import time
import shutil
import tempfile
from zipfile import ZipFile
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class FaceMatcher:
    """Image-based face recognition system that finds matches in a database"""
    
    def __init__(self, model_name='buffalo_s', images_folder="images", threshold=0.4):
        """
        Initialize the face matcher.
        
        Args:
            model_name (str): InsightFace model to use
            images_folder (str): Path to folder containing reference images
            threshold (float): Similarity threshold for matching
        """
        print("Initializing InsightFace...")
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.images_folder = images_folder
        self.threshold = threshold
        
        # Database of known faces
        self.known_faces: Dict[str, Dict] = {}
        
        # Load the database
        self.load_face_database()
        
    def load_face_database(self):
        """Load all faces from the images folder into the database"""
        print(f"Loading face database from '{self.images_folder}'...")
        
        if not os.path.exists(self.images_folder):
            print(f"Warning: Images folder '{self.images_folder}' not found!")
            return
            
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        loaded_count = 0
        
        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(self.images_folder, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    # Load image
                    image = cv2.imread(filepath)
                    if image is None:
                        print(f"Warning: Could not load image: {filename}")
                        continue
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the image
                    faces = self.app.get(image_rgb)
                    
                    if len(faces) > 0:
                        # Use the first face found
                        face = faces[0]
                        embedding = face.embedding
                        
                        # Store face data
                        self.known_faces[name] = {
                            'embedding': embedding,
                            'image_path': filepath,
                            'filename': filename,
                            'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None
                        }
                        
                        loaded_count += 1
                        print(f"  ✓ Loaded: {name} from {filename}")
                    else:
                        print(f"  ✗ No face found in: {filename}")
                        
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
        
        print(f"Successfully loaded {loaded_count} faces into database.")
        print("-" * 50)
    
    def extract_faces_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract all faces from an input image.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            List of face dictionaries with embeddings and bounding boxes
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(image_rgb)
        
        extracted_faces = []
        for i, face in enumerate(faces):
            face_data = {
                'index': i,
                'embedding': face.embedding,
                'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                'confidence': getattr(face, 'det_score', 1.0)
            }
            extracted_faces.append(face_data)
        
        return extracted_faces
    
    def find_matches(self, face_embedding: np.ndarray) -> List[Tuple[str, float, Dict]]:
        """
        Find matching faces in the database for a given embedding.
        
        Args:
            face_embedding: Face embedding to match
            
        Returns:
            List of (name, similarity_score, face_data) tuples, sorted by similarity
        """
        if not self.known_faces:
            return []
        
        matches = []
        
        for name, face_data in self.known_faces.items():
            known_embedding = face_data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, known_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
            )
            
            # Only include matches above threshold
            if similarity >= self.threshold:
                matches.append((name, similarity, face_data))
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process an input image and find all matching faces.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            Dictionary with processing results
        """
        print(f"Processing image: {image_path}")
        
        start_time = time.time()
        
        try:
            # Extract faces from input image
            input_faces = self.extract_faces_from_image(image_path)
            
            if not input_faces:
                return {
                    'success': False,
                    'error': 'No faces detected in input image',
                    'processing_time': time.time() - start_time
                }
            
            print(f"Found {len(input_faces)} face(s) in input image")
            
            # Find matches for each face
            all_matches = []
            
            for face_data in input_faces:
                face_index = face_data['index']
                face_embedding = face_data['embedding']
                
                matches = self.find_matches(face_embedding)
                
                face_result = {
                    'face_index': face_index,
                    'bbox': face_data['bbox'],
                    'detection_confidence': face_data['confidence'],
                    'matches': []
                }
                
                for name, similarity, known_face_data in matches:
                    match_info = {
                        'name': name,
                        'similarity_score': float(similarity),
                        'accuracy_percentage': round(similarity * 100, 2),
                        'matched_image_path': known_face_data['image_path'],
                        'matched_filename': known_face_data['filename']
                    }
                    face_result['matches'].append(match_info)
                
                all_matches.append(face_result)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'input_image': image_path,
                'faces_detected': len(input_faces),
                'processing_time': round(processing_time, 3),
                'database_size': len(self.known_faces),
                'threshold_used': self.threshold,
                'results': all_matches
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def print_results(self, results: Dict):
        """Print results in a formatted way"""
        if not results['success']:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n📊 Processing Results:")
        print(f"   Input Image: {results['input_image']}")
        print(f"   Faces Detected: {results['faces_detected']}")
        print(f"   Database Size: {results['database_size']} faces")
        print(f"   Processing Time: {results['processing_time']}s")
        print(f"   Threshold: {results['threshold_used']}")
        print("-" * 60)
        
        for i, face_result in enumerate(results['results']):
            print(f"\n👤 Face #{face_result['face_index'] + 1}:")
            
            if not face_result['matches']:
                print("   ❌ No matches found (below threshold)")
                continue
            
            print(f"   📍 Detection Confidence: {face_result['detection_confidence']:.2f}")
            print(f"   🎯 Matches Found: {len(face_result['matches'])}")
            
            for j, match in enumerate(face_result['matches']):
                print(f"\n   #{j+1} Match:")
                print(f"      👤 Name: {match['name']}")
                print(f"      📊 Similarity: {match['similarity_score']:.4f}")
                print(f"      🎯 Accuracy: {match['accuracy_percentage']}%")
                print(f"      📁 File: {match['matched_filename']}")
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to a JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Face Matcher - Find matching faces in image database')
    parser.add_argument('input_image', help='Path to input image to analyze')
    parser.add_argument('--images-folder', '-d', default='images', 
                       help='Folder containing reference images (default: images)')
    parser.add_argument('--threshold', '-t', type=float, default=0.4,
                       help='Similarity threshold (default: 0.4)')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--zip', help='Output zip file containing all matched images')
    parser.add_argument('--model', '-m', default='buffalo_s',
                       help='InsightFace model to use (default: buffalo_s)')
    
    args = parser.parse_args()
    
    # Initialize face matcher
    matcher = FaceMatcher(
        model_name=args.model,
        images_folder=args.images_folder,
        threshold=args.threshold
    )
    
    # Process the input image
    results = matcher.process_image(args.input_image)
    
    # Print results
    matcher.print_results(results)
    
    # Save results if requested
    if args.output:
        matcher.save_results(results, args.output)

    # Save matched images to zip if requested
    if args.zip and results['success']:
        # Collect all unique matched image paths
        matched_paths = set()
        for face_result in results['results']:
            for match in face_result['matches']:
                matched_paths.add(match['matched_image_path'])
        if not matched_paths:
            print(f"No matches found, so no zip file will be created.")
            return
        # Create a temp directory to copy images
        with tempfile.TemporaryDirectory() as tempdir:
            for img_path in matched_paths:
                if os.path.exists(img_path):
                    shutil.copy(img_path, tempdir)
            # Create the zip file
            with ZipFile(args.zip, 'w') as zipf:
                for img_file in os.listdir(tempdir):
                    zipf.write(os.path.join(tempdir, img_file), arcname=img_file)
            print(f"\n🗜️  Matched images zipped to: {args.zip}")

if __name__ == "__main__":
    main() 