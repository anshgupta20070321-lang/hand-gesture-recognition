import cv2
import numpy as np
import os
from collections import deque
from keras.models import load_model
from image_processing import preprocess_image

# Configuration
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_FRAMES = 5  # Number of frames to average predictions

# Gesture labels
GESTURES = {
    0: 'UP',
    1: 'DOWN',
    2: 'FORWARD',
    3: 'BACKWARD',
    4: 'RIGHT',
    5: 'LEFT',
    6: 'LAND'
}

# Colors for each gesture (BGR format)
GESTURE_COLORS = {
    0: (0, 255, 0),      # Green - UP
    1: (0, 0, 255),      # Red - DOWN
    2: (255, 0, 0),      # Blue - FORWARD
    3: (0, 255, 255),    # Yellow - BACKWARD
    4: (255, 0, 255),    # Magenta - RIGHT
    5: (128, 0, 128),    # Purple - LEFT
    6: (0, 128, 255)     # Orange - LAND
}


class GesturePredictor:
    def __init__(self, model_path='gesture_model_best.keras'):
        """Initialize the gesture predictor"""
        self.model_path = model_path
        self.model = None
        self.prediction_history = deque(maxlen=SMOOTHING_FRAMES)
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = load_model(self.model_path)
        print("Model loaded successfully!")
        
        # Print model info
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for prediction"""
        # Preprocess image
        processed = preprocess_image(frame, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Normalize
        processed = processed.astype('float32') / 255.0
        
        # Add channel and batch dimensions
        processed = np.expand_dims(processed, axis=-1)  # Add channel
        processed = np.expand_dims(processed, axis=0)   # Add batch
        
        return processed
    
    def predict(self, frame, use_smoothing=True):
        """
        Predict gesture from frame
        
        Args:
            frame: Input frame (ROI)
            use_smoothing: Whether to use temporal smoothing
        
        Returns:
            gesture_id, confidence, all_probabilities
        """
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        if use_smoothing:
            # Add to history
            self.prediction_history.append(predictions)
            
            # Average predictions over recent frames
            smoothed_predictions = np.mean(self.prediction_history, axis=0)
            predictions = smoothed_predictions
        
        # Get top prediction
        gesture_id = np.argmax(predictions)
        confidence = predictions[gesture_id]
        
        return gesture_id, confidence, predictions
    
    def draw_info(self, frame, gesture_id, confidence, all_predictions):
        """Draw prediction information on frame"""
        h, w = frame.shape[:2]
        
        # Create info panel
        panel_height = 250
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Get gesture name and color
        gesture_name = GESTURES.get(gesture_id, 'UNKNOWN')
        color = GESTURE_COLORS.get(gesture_id, (255, 255, 255))
        
        # Draw main prediction
        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(panel, f"GESTURE: {gesture_name}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 3)
            cv2.putText(panel, f"Confidence: {confidence*100:.1f}%",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, color, 2)
        else:
            cv2.putText(panel, "NO CLEAR GESTURE",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (128, 128, 128), 2)
            cv2.putText(panel, f"Confidence: {confidence*100:.1f}% (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (128, 128, 128), 1)
        
        # Draw probability bars for all gestures
        y_offset = 120
        bar_width = w - 200
        bar_height = 15
        
        for i, prob in enumerate(all_predictions):
            gesture_name = GESTURES.get(i, f'Class {i}')
            gesture_color = GESTURE_COLORS.get(i, (255, 255, 255))
            
            # Draw label
            cv2.putText(panel, f"{gesture_name}:",
                       (10, y_offset), cv2.FONT_HERSHEY_PLAIN,
                       0.9, (200, 200, 200), 1)
            
            # Draw probability bar
            bar_x = 100
            filled_width = int(bar_width * prob)
            
            # Background bar
            cv2.rectangle(panel,
                         (bar_x, y_offset - 10),
                         (bar_x + bar_width, y_offset + 2),
                         (50, 50, 50), -1)
            
            # Filled bar
            if filled_width > 0:
                cv2.rectangle(panel,
                             (bar_x, y_offset - 10),
                             (bar_x + filled_width, y_offset + 2),
                             gesture_color, -1)
            
            # Probability text
            cv2.putText(panel, f"{prob*100:.1f}%",
                       (bar_x + bar_width + 10, y_offset),
                       cv2.FONT_HERSHEY_PLAIN,
                       0.8, (200, 200, 200), 1)
            
            y_offset += 18
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        
        return combined
    
    def run_realtime(self):
        """Run real-time gesture prediction"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("REAL-TIME GESTURE PREDICTION")
        print("="*60)
        print("\nControls:")
        print("  ESC - Quit")
        print("  SPACE - Pause/Resume")
        print("  'r' - Reset prediction history")
        print("  's' - Save current frame")
        print("="*60)
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Mirror image
                frame = cv2.flip(frame, 1)
                
                # Draw ROI rectangle
                cv2.rectangle(frame, (219, 9), (621, 411), (0, 255, 0), 2)
                
                # Extract ROI
                roi = frame[10:410, 220:520]
                
                # Predict gesture
                gesture_id, confidence, all_predictions = self.predict(roi)
                
                # Draw information
                display_frame = self.draw_info(frame.copy(), gesture_id, 
                                              confidence, all_predictions)
                
                # Show processed image
                processed_display = preprocess_image(roi, target_size=(200, 200))
                cv2.imshow("Processed ROI", processed_display)
                
                frame_count += 1
            else:
                # Show paused message
                paused_frame = display_frame.copy()
                cv2.putText(paused_frame, "PAUSED", 
                           (paused_frame.shape[1]//2 - 100, paused_frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                display_frame = paused_frame
            
            # Display
            cv2.imshow("Gesture Recognition", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # Reset
                self.prediction_history.clear()
                print("Prediction history reset")
            elif key == ord('s'):  # Save
                filename = f"gesture_capture_{frame_count}.jpg"
                cv2.imwrite(filename, roi)
                print(f"Saved frame: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print(f"Total frames processed: {frame_count}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time gesture prediction')
    parser.add_argument('--model', default='gesture_model_best.keras',
                       help='Path to trained model (default: gesture_model_best.keras)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold (default: 0.6)')
    parser.add_argument('--smoothing', type=int, default=5,
                       help='Number of frames for smoothing (default: 5)')
    
    args = parser.parse_args()
    
    # Update global config
    global CONFIDENCE_THRESHOLD, SMOOTHING_FRAMES
    CONFIDENCE_THRESHOLD = args.threshold
    SMOOTHING_FRAMES = args.smoothing
    
    # Create predictor
    try:
        predictor = GesturePredictor(args.model)
        predictor.run_realtime()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nAvailable model files:")
        for f in os.listdir('.'):
            if f.endswith(('.keras', '.h5')):
                print(f"  - {f}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
