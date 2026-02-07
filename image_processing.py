import cv2
import numpy as np

MIN_VALUE = 70

def preprocess_image(image_path_or_array, target_size=(128, 128)):
    """
    Preprocess image for training/prediction
    
    Args:
        image_path_or_array: Either path to image or numpy array
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed grayscale image
    """
    # Load image if path is provided
    if isinstance(image_path_or_array, str):
        frame = cv2.imread(image_path_or_array)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path_or_array}")
    else:
        frame = image_path_or_array
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    
    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 
        2
    )
    
    # Otsu's thresholding for better binary image
    _, binary = cv2.threshold(
        adaptive_thresh,
        MIN_VALUE,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Resize to target size
    if target_size is not None:
        binary = cv2.resize(binary, target_size)
    
    return binary


def preprocess_for_model(image, normalize=True):
    """
    Prepare preprocessed image for model input
    
    Args:
        image: Preprocessed grayscale image
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Image ready for model (with batch and channel dimensions)
    """
    # Normalize
    if normalize:
        image = image.astype('float32') / 255.0
    
    # Add channel dimension (grayscale)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    return image


def enhance_hand_region(image):
    """
    Additional enhancement for hand region detection
    
    Args:
        image: Input image (grayscale)
    
    Returns:
        Enhanced image
    """
    # Histogram equalization
    enhanced = cv2.equalizeHist(image)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    
    return enhanced


# Backward compatibility with old function name
def func(path):
    """Legacy function for compatibility"""
    return preprocess_image(path, target_size=None)


if __name__ == "__main__":
    # Test the preprocessing
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        
        # Test preprocessing
        processed = preprocess_image(test_image_path)
        
        # Display
        cv2.imshow("Original", cv2.imread(test_image_path))
        cv2.imshow("Processed", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Usage: python image_processing_improved.py <image_path>")

