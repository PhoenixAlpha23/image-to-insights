def compare_detections(before_objects, after_objects):
    """
    Compares two lists of YOLO detections and returns a list of textual differences.
    """
    def to_label_set(objects):
        return set([obj['class'] for obj in objects])

    before_set = to_label_set(before_objects)
    after_set = to_label_set(after_objects)

    added = after_set - before_set
    removed = before_set - after_set

    description = []

    if added:
        for item in added:
            description.append(f"A new {item} appeared.")
    if removed:
        for item in removed:
            description.append(f"A {item} was removed.")

    if not description:
        description.append("No major object-level changes detected.")

    return " ".join(description)
