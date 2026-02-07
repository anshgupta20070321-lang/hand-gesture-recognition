import cv2
import numpy as np
import os
mode = 'test'  

GESTURES = {
    '0': 'up',
    '1': 'down', 
    '2': 'forward',
    '3': 'backward',
    '4': 'right',
    '5': 'left',
    '6': 'land'
}

IMG_SIZE = 128
MIN_THRESHOLD = 70

class DataCollector:
    def __init__(self, mode='train'):
        self.mode = mode
        self.directory = f'data/{mode}/'
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure for data collection"""
        for folder in ['data', f'data/{self.mode}']:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        for gesture_id in GESTURES.keys():
            path = f'data/{self.mode}/{gesture_id}'
            if not os.path.exists(path):
                os.makedirs(path)
    
    def get_counts(self):
        """Get current image counts for each gesture"""
        counts = {}
        for gesture_id, gesture_name in GESTURES.items():
            path = f'{self.directory}{gesture_id}'
            counts[gesture_name] = len(os.listdir(path))
        return counts
    
    def preprocess_roi(self, roi):
        """Apply preprocessing to ROI (same as will be used in training)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Adaptive thresholding
        th = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Otsu thresholding
        _, processed = cv2.threshold(
            th, MIN_THRESHOLD, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        return processed
    
    def draw_info(self, frame, counts):
        """Draw information overlay on frame"""
        # Mode
        cv2.putText(frame, f"MODE: {self.mode.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Counts
        y_offset = 60
        for gesture_id, gesture_name in GESTURES.items():
            text = f"{gesture_id}: {gesture_name} = {counts[gesture_name]}"
            cv2.putText(frame, text,
                       (10, y_offset), cv2.FONT_HERSHEY_PLAIN,
                       1, (0, 255, 255), 1)
            y_offset += 20
        
        # Instructions
        cv2.putText(frame, "Press 0-6 to capture | ESC to quit",
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_PLAIN, 
                   1, (255, 255, 255), 1)
        
        # ROI rectangle
        cv2.rectangle(frame, (219, 9), (621, 411), (0, 255, 0), 2)
    
    def run(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Starting data collection in {self.mode} mode...")
        print("Press 0-6 to save images for each gesture")
        print("Press ESC to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Mirror image
            frame = cv2.flip(frame, 1)
            
            # Get current counts
            counts = self.get_counts()
            
            # Extract ROI
            roi = frame[10:410, 220:520]
            
            # Process ROI for preview
            processed = self.preprocess_roi(roi)
            processed_resized = cv2.resize(processed, (200, 200))
            
            # Draw information
            self.draw_info(frame, counts)
            
            # Display frames
            cv2.imshow("Data Collection", frame)
            cv2.imshow("Processed Preview", processed_resized)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            # Save image for corresponding gesture
            for gesture_id in GESTURES.keys():
                if key == ord(gesture_id):
                    count = counts[GESTURES[gesture_id]]
                    filename = f'{self.directory}{gesture_id}/{count}.jpg'
                    cv2.imwrite(filename, roi)
                    print(f"Saved {GESTURES[gesture_id]}: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*50)
        print("Data Collection Summary:")
        print("="*50)
        final_counts = self.get_counts()
        for gesture_name, count in final_counts.items():
            print(f"{gesture_name:12s}: {count:4d} images")
        print("="*50)


if __name__ == "__main__":
    collector = DataCollector(mode=mode)
    collector.run()
