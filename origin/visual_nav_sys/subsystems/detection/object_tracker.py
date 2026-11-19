"""
Object Detection and Tracking
YOLOv11-based object detection and guidance estimation
"""

import numpy as np


def extract_object_info(results):
    """Extract object information from YOLO results"""
    objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            objects.append(
                {
                    "class": result.names[cls],
                    "center": (center_x, center_y),
                    "area": area,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

    return objects


def estimate_guidance_from_objects(ref_objects, curr_objects):
    """
    Estimate camera guidance based on detected object positions
    Returns: (dx, dy, dz_scale, confidence, found)
    """
    if len(ref_objects) == 0 or len(curr_objects) == 0:
        return 0, 0, 1.0, 0.0, False

    matched_pairs = []
    for ref_obj in ref_objects:
        ref_class = ref_obj["class"]
        ref_center = ref_obj["center"]
        ref_area = ref_obj["area"]

        for curr_obj in curr_objects:
            if curr_obj["class"] == ref_class:
                matched_pairs.append(
                    {
                        "ref_center": ref_center,
                        "curr_center": curr_obj["center"],
                        "ref_area": ref_area,
                        "curr_area": curr_obj["area"],
                    }
                )
                break

    if len(matched_pairs) == 0:
        return 0, 0, 1.0, 0.0, False

    dx_total, dy_total, scale_total = 0, 0, 0
    for pair in matched_pairs:
        dx_total += pair["ref_center"][0] - pair["curr_center"][0]
        dy_total += pair["ref_center"][1] - pair["curr_center"][1]
        scale_total += np.sqrt(pair["ref_area"] / (pair["curr_area"] + 1e-6))

    n = len(matched_pairs)
    dx = dx_total / n
    dy = dy_total / n
    scale = scale_total / n

    confidence = min(100, n * 30)

    return dx, dy, scale, confidence, True
