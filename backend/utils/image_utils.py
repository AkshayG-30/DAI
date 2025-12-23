import cv2
import numpy as np
from PIL import Image

def segment_hieroglyphs(image_path):
    """Segment hieroglyphs from image using OpenCV"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h_img, w_img = th.shape
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter small areas and full-image contours
            if area < 200:
                continue
            if w > 0.95*w_img or h > 0.95*h_img:
                continue
                
            boxes.append((x, y, w, h))
        
        # If no boxes found, return full image
        if not boxes:
            return [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
        
        # Sort boxes by position (top to bottom, left to right)
        boxes = sorted(boxes, key=lambda b: (b[1]//50, b[0]))
        
        # Extract crops
        crops = []
        for (x, y, w, h) in boxes:
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w_img, x + w + pad)
            y1 = min(h_img, y + h + pad)
            
            crop = img[y0:y1, x0:x1]
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        
        return crops
        
    except Exception as e:
        print(f"[ERROR] Hieroglyph segmentation failed: {e}")
        return []

def validate_image(file):
    """Validate uploaded image file"""
    from config import Config
    config = Config()
    
    # Check file size
    if hasattr(file, 'content_length') and file.content_length > config.MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size: {config.MAX_FILE_SIZE} bytes")
    
    # Check file extension
    if not file.filename or '.' not in file.filename:
        raise ValueError("Invalid filename")
    
    extension = file.filename.rsplit('.', 1)[1].lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}")
    
    # Try to open as image
    try:
        image = Image.open(file.stream)
        image.verify()
        file.stream.seek(0)  # Reset stream for later use
        return True
    except Exception:
        raise ValueError("File is not a valid image")

def preprocess_for_latin_ocr(image_path):
    """Specialized preprocessing for Latin texts"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding for varying lighting
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
        
    except Exception as e:
        print(f"[ERROR] Latin preprocessing failed: {e}")
        return None

def enhance_contrast_for_manuscripts(image):
    """Enhanced contrast specifically for manuscript images"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced