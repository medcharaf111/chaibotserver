#!/usr/bin/env python3
"""
Face Matcher GUI - Image-based Face Recognition with Graphical Interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
import time
from typing import List, Dict, Tuple
from PIL import Image, ImageTk
import threading
import shutil
import tempfile
from zipfile import ZipFile

class FaceMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Matcher - Find Matching Faces")
        self.root.geometry("1000x700")
        
        # Initialize face recognition
        self.app = None
        self.known_faces = {}
        self.images_folder = "images"
        self.threshold = 0.4
        
        # GUI variables
        self.input_image_path = tk.StringVar()
        self.results_text = tk.StringVar()
        self.processing = False
        
        self.setup_ui()
        self.load_face_database()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Matcher", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input image selection
        ttk.Label(main_frame, text="Input Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_image_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=1, column=2, padx=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Threshold setting
        ttk.Label(settings_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.threshold_var, 
                                   orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=10)
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.threshold:.2f}")
        self.threshold_label.grid(row=0, column=2)
        threshold_scale.configure(command=self.update_threshold_label)
        
        # Database info
        self.db_info_label = ttk.Label(settings_frame, text="Database: Loading...")
        self.db_info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Find Matches", command=self.process_image)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text widget for results
        self.results_text_widget = tk.Text(results_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text_widget.yview)
        self.results_text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.results_text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Save results button
        self.save_button = ttk.Button(main_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.grid(row=6, column=0, pady=5)
        # Save zip button
        self.save_zip_button = ttk.Button(main_frame, text="Save Matched Images as Zip", command=self.save_zip, state=tk.DISABLED)
        self.save_zip_button.grid(row=6, column=1, columnspan=2, pady=5)
        
        self.results_data = None
    
    def update_threshold_label(self, value):
        """Update threshold label when scale changes"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"{self.threshold:.2f}")
    
    def browse_image(self):
        """Browse for input image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.input_image_path.set(file_path)
    
    def load_face_database(self):
        """Load face database in a separate thread"""
        def load_db():
            try:
                self.db_info_label.config(text="Database: Loading...")
                
                # Initialize InsightFace
                self.app = FaceAnalysis(name='buffalo_s')
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                
                if not os.path.exists(self.images_folder):
                    self.db_info_label.config(text="Database: No images folder found!")
                    return
                
                supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                loaded_count = 0
                
                for filename in os.listdir(self.images_folder):
                    if filename.lower().endswith(supported_formats):
                        filepath = os.path.join(self.images_folder, filename)
                        name = os.path.splitext(filename)[0]
                        
                        try:
                            image = cv2.imread(filepath)
                            if image is None:
                                continue
                            
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            faces = self.app.get(image_rgb)
                            
                            if len(faces) > 0:
                                face = faces[0]
                                self.known_faces[name] = {
                                    'embedding': face.embedding,
                                    'image_path': filepath,
                                    'filename': filename
                                }
                                loaded_count += 1
                                
                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
                
                self.db_info_label.config(text=f"Database: {loaded_count} faces loaded")
                
            except Exception as e:
                self.db_info_label.config(text=f"Database: Error loading - {str(e)}")
        
        # Run in separate thread to avoid blocking GUI
        threading.Thread(target=load_db, daemon=True).start()
    
    def process_image(self):
        """Process the input image to find matches"""
        if not self.input_image_path.get():
            messagebox.showerror("Error", "Please select an input image")
            return
        
        if not self.app:
            messagebox.showerror("Error", "Face recognition system not ready yet")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.progress.start()
        
        # Clear previous results
        self.results_text_widget.delete(1.0, tk.END)
        self.save_button.config(state=tk.DISABLED)
        self.save_zip_button.config(state=tk.DISABLED)
        
        def process():
            try:
                results = self.process_image_worker(self.input_image_path.get())
                self.root.after(0, self.display_results, results)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.processing_finished)
        
        threading.Thread(target=process, daemon=True).start()
    
    def process_image_worker(self, image_path):
        """Worker function to process image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(image_rgb)
        
        if not faces:
            return {
                'success': False,
                'error': 'No faces detected in input image'
            }
        
        # Find matches for each face
        all_matches = []
        
        for i, face in enumerate(faces):
            face_embedding = face.embedding
            matches = []
            
            for name, face_data in self.known_faces.items():
                known_embedding = face_data['embedding']
                
                # Calculate cosine similarity
                similarity = np.dot(face_embedding, known_embedding) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                )
                
                if similarity >= self.threshold:
                    matches.append({
                        'name': name,
                        'similarity_score': float(similarity),
                        'accuracy_percentage': round(similarity * 100, 2),
                        'matched_filename': face_data['filename']
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            all_matches.append({
                'face_index': i,
                'matches': matches
            })
        
        return {
            'success': True,
            'faces_detected': len(faces),
            'results': all_matches
        }
    
    def display_results(self, results):
        """Display results in the text widget"""
        if not results['success']:
            self.results_text_widget.insert(tk.END, f"❌ Error: {results['error']}\n")
            return
        
        self.results_text_widget.insert(tk.END, f"📊 Processing Results:\n")
        self.results_text_widget.insert(tk.END, f"   Input Image: {self.input_image_path.get()}\n")
        self.results_text_widget.insert(tk.END, f"   Faces Detected: {results['faces_detected']}\n")
        self.results_text_widget.insert(tk.END, f"   Database Size: {len(self.known_faces)} faces\n")
        self.results_text_widget.insert(tk.END, f"   Threshold: {self.threshold:.2f}\n")
        self.results_text_widget.insert(tk.END, "-" * 60 + "\n\n")
        
        for face_result in results['results']:
            self.results_text_widget.insert(tk.END, f"👤 Face #{face_result['face_index'] + 1}:\n")
            
            if not face_result['matches']:
                self.results_text_widget.insert(tk.END, "   ❌ No matches found (below threshold)\n\n")
                continue
            
            self.results_text_widget.insert(tk.END, f"   🎯 Matches Found: {len(face_result['matches'])}\n")
            
            for j, match in enumerate(face_result['matches']):
                self.results_text_widget.insert(tk.END, f"\n   #{j+1} Match:\n")
                self.results_text_widget.insert(tk.END, f"      👤 Name: {match['name']}\n")
                self.results_text_widget.insert(tk.END, f"      📊 Similarity: {match['similarity_score']:.4f}\n")
                self.results_text_widget.insert(tk.END, f"      🎯 Accuracy: {match['accuracy_percentage']}%\n")
                self.results_text_widget.insert(tk.END, f"      📁 File: {match['matched_filename']}\n")
            
            self.results_text_widget.insert(tk.END, "\n")
        
        self.results_data = results
        self.save_button.config(state=tk.NORMAL)
        self.save_zip_button.config(state=tk.NORMAL)
    
    def processing_finished(self):
        """Called when processing is finished"""
        self.processing = False
        self.process_button.config(state=tk.NORMAL)
        self.progress.stop()
    
    def save_results(self):
        """Save results to JSON file"""
        if not self.results_data:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.results_data, f, indent=2)
                messagebox.showinfo("Success", f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def save_zip(self):
        """Save all matched images as a zip file"""
        if not self.results_data or not self.results_data.get('success'):
            messagebox.showerror("Error", "No results to save.")
            return
        # Collect all unique matched image paths
        matched_paths = set()
        for face_result in self.results_data['results']:
            for match in face_result['matches']:
                matched_paths.add(match['matched_filename'])
        if not matched_paths:
            messagebox.showinfo("No Matches", "No matched images to save.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Matched Images as Zip",
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with tempfile.TemporaryDirectory() as tempdir:
                    for img_name in matched_paths:
                        img_path = os.path.join(self.images_folder, img_name)
                        if os.path.exists(img_path):
                            shutil.copy(img_path, tempdir)
                    with ZipFile(file_path, 'w') as zipf:
                        for img_file in os.listdir(tempdir):
                            zipf.write(os.path.join(tempdir, img_file), arcname=img_file)
                messagebox.showinfo("Success", f"Matched images zipped to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create zip: {str(e)}")

def main():
    root = tk.Tk()
    app = FaceMatcherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 