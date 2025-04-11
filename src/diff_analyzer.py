def compare_detections(before_objects, after_objects):
    """
    Compares two lists of YOLO detections and returns a list of textual differences.
    """
    # Count objects by class
    before_counts = {}
    after_counts = {}
    
    # Apply confidence threshold
    threshold = 0.6  # Filter out low confidence detections
    
    # Filter and count objects
    for obj in before_objects:
        if obj['confidence'] >= threshold:
            before_counts[obj['class']] = before_counts.get(obj['class'], 0) + 1
    
    for obj in after_objects:
        if obj['confidence'] >= threshold:
            after_counts[obj['class']] = after_counts.get(obj['class'], 0) + 1
    
    # Find differences
    all_classes = set(list(before_counts.keys()) + list(after_counts.keys()))
    description = []
    
    for cls in all_classes:
        before_count = before_counts.get(cls, 0)
        after_count = after_counts.get(cls, 0)
        
        if before_count < after_count:
            diff = after_count - before_count
            if before_count == 0:
                description.append(f"A {cls} appeared.")
            else:
                description.append(f"{diff} additional {cls}(s) appeared.")
        elif before_count > after_count:
            diff = before_count - after_count
            if after_count == 0:
                description.append(f"A {cls} was removed.")
            else:
                description.append(f"{diff} {cls}(s) were removed.")
    
    if not description:
        description.append("No significant changes detected between the images.")
    
    return " ".join(description)
