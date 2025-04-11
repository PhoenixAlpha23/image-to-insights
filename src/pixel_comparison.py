import cv2
import numpy as np
from PIL import Image

def detect_pixel_differences(before_img, after_img, threshold=30, min_area=50):
    """
    Uses OpenCV to detect pixel-level differences between images
    
    Args:
        before_img: PIL Image of the "before" image
        after_img: PIL Image of the "after" image
        threshold: Sensitivity threshold (higher = less sensitive)
        min_area: Minimum size of differences to consider
        
    Returns:
        changes_detected: Boolean indicating if changes were found
        change_areas: Number of distinct change areas found
        diff_image: PIL Image with differences highlighted
    """
    # Convert PIL images to cv2 format
    before_cv = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2BGR)
    after_cv = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2BGR)
    
    # Make sure images are the same size
    before_cv = cv2.resize(before_cv, (640, 480))
    after_cv = cv2.resize(after_cv, (640, 480))
    
    # Calculate absolute difference between images
    diff = cv2.absdiff(before_cv, after_cv)
    
    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by size (ignore very small differences)
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Create visual representation
    diff_image = after_cv.copy()
    cv2.drawContours(diff_image, significant_contours, -1, (0, 0, 255), 2)
    
    # Create PIL image for Streamlit
    diff_pil = Image.fromarray(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
    
    if significant_contours:
        changes_detected = True
        change_areas = len(significant_contours)
    else:
        changes_detected = False
        change_areas = 0
        
    return changes_detected, change_areas, diff_pil
