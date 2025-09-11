import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple

class TrafficMonitor:
    """YOLO-based vehicle detection for traffic monitoring"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the traffic monitoring system
        
        Args:
            model_path (str): Path to YOLO model weights
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.vehicle_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # Detection zones for each direction (x, y, width, height)
        self.detection_zones = {
            'north': (200, 50, 200, 150),   # Top section
            'south': (200, 350, 200, 150),  # Bottom section
            'east': (400, 200, 150, 150),   # Right section  
            'west': (50, 200, 150, 150)     # Left section
        }
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7, 8]  # car, motorbike, bus, truck, train
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in the given frame
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            List[Dict]: List of detected vehicles with bounding boxes and confidence
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for vehicle classes only
                    class_id = int(box.cls[0])
                    if class_id in self.vehicle_classes:
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Apply confidence threshold
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': (x1, y1, x2-x1, y2-y1),  # x, y, width, height
                                'confidence': confidence,
                                'class_id': class_id,
                                'center': ((x1+x2)//2, (y1+y2)//2)
                            })
        
        return detections
    
    def count_vehicles_by_direction(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles in each direction based on detection zones
        
        Args:
            detections (List[Dict]): Vehicle detections from YOLO
            
        Returns:
            Dict[str, int]: Vehicle count for each direction
        """
        counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Check which zone the vehicle center falls into
            for direction, (zone_x, zone_y, zone_w, zone_h) in self.detection_zones.items():
                if (zone_x <= center_x <= zone_x + zone_w and 
                    zone_y <= center_y <= zone_y + zone_h):
                    counts[direction] += 1
                    break  # Each vehicle counted only once
        
        # Update internal state
        self.vehicle_counts = counts
        return counts
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and information on frame
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): Vehicle detections
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        # Create a copy to avoid modifying original frame
        display_frame = frame.copy()
        
        # Draw detection zones
        zone_colors = {
            'north': (0, 255, 255),    # Yellow
            'south': (255, 0, 255),    # Magenta  
            'east': (0, 255, 0),       # Green
            'west': (255, 0, 0)        # Blue
        }
        
        for direction, (x, y, w, h) in self.detection_zones.items():
            color = zone_colors[direction]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Add direction label and count
            label = f"{direction.upper()}: {self.vehicle_counts[direction]}"
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw vehicle detections
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence score
            conf_label = f"{confidence:.2f}"
            cv2.putText(display_frame, conf_label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(display_frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Add FPS counter
        self.update_fps()
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add total vehicle count
        total_vehicles = sum(self.vehicle_counts.values())
        total_text = f"Total Vehicles: {total_vehicles}"
        cv2.putText(display_frame, total_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display_frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def process_video_stream(self, source: str, display: bool = True) -> None:
        """
        Process video stream for real-time detection
        
        Args:
            source (str): Video source (file path, camera index, or RTSP URL)
            display (bool): Whether to display the video with detections
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print(f"Processing video stream from: {source}")
        print("Press 'q' to quit, 's' to save screenshot")
        
        screenshot_counter = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream or failed to read frame")
                break
            
            # Resize frame for better performance (optional)
            # frame = cv2.resize(frame, (640, 480))
            
            # Detect vehicles
            detections = self.detect_vehicles(frame)
            
            # Count vehicles by direction
            vehicle_counts = self.count_vehicles_by_direction(detections)
            
            # Draw detections if display is enabled
            if display:
                display_frame = self.draw_detections(frame, detections)
                cv2.imshow('AI Traffic Control - Vehicle Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"traffic_screenshot_{screenshot_counter:04d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
                    screenshot_counter += 1
            
            # Print current counts (optional)
            print(f"Current counts: {vehicle_counts}", end='\r')
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dict: Performance metrics
        """
        return {
            'fps': self.current_fps,
            'confidence_threshold': self.confidence_threshold,
            'total_vehicles': sum(self.vehicle_counts.values()),
            'vehicle_counts': self.vehicle_counts.copy(),
            'model_loaded': self.model is not None
        }

def test_traffic_monitor():
    """Test function for TrafficMonitor"""
    # Initialize traffic monitor
    monitor = TrafficMonitor()
    
    # Test with webcam (use 0 for default camera)
    print("Testing with webcam...")
    print("Make sure your camera is connected and working")
    
    # You can also test with a video file:
    # monitor.process_video_stream('path_to_your_video.mp4')
    
    monitor.process_video_stream(0)  # Use webcam

if __name__ == "__main__":
    test_traffic_monitor()
