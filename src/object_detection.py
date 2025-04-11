from ultralytics import YOLO
import numpy as np
from PIL import Image

model = YOLO("yolov8s.pt") 

def detect_objects(pil_image, conf_threshold=0.6):
    """
    Runs YOLOv8 on a PIL image and returns detected objects.
    Returns: 
    - List of dicts with class name, confidence, and bounding box
    - Annotated image     
    """
    img_array = np.array(pil_image.convert("RGB"))
    
    # inference with confidence threshold
    results = model(img_array, conf=conf_threshold, verbose=False)
    objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            name = result.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append({
                "class": name,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })
    annotated_array = results[0].plot()
    annotated_image = Image.fromarray(annotated_array)
    return objects, annotated_image
