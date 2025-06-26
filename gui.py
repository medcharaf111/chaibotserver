import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FaceRecognitionGUI:
    """Simple GUI for the face recognition system"""
    
    def __init__(self, recognition_system):
        self.recognition_system = recognition_system
        
    def draw_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw recognition results on frame"""
        for i, result in enumerate(results):
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
        
        # Add database info
        db_count = self.recognition_system.get_database_count()
        cv2.putText(frame, f"Database: {db_count} faces", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save current frame"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 60 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def handle_keypress(self, key: int, frame: np.ndarray, results: List[Dict[str, Any]]) -> bool:
        """Handle keypress events"""
        if key == ord('q'):
            return False  # Quit
        elif key == ord('s'):
            # Save current frame
            import time
            timestamp = int(time.time())
            save_path = f"frame_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)
            logger.info(f"Frame saved as: {save_path}")
        
        return True  # Continue 