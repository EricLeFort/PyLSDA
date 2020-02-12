def non_max_suppression(scored_boxes, max_overlap):
    """
    Greedily select high-scoring detections and skip detections that are significantly covered by
    a previously selected detection.

    NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    but an inner loop has been eliminated to significantly speed it
    up in the case of a large number of boxes

    Args:
        boxes (list of (Box, score)): The scored boxes
        max_overlap (number): The maximum allowed overlap
    Returns:
        The selections
    """
    if not scored_boxes:
        return []

    areas = [(box[2]-box[0]+1) * (box[3]-box[1]+1) for box, _ in scored_boxes]

    # Sort by score, record original index
    decorated_boxes = [(box, score, i) for i, (box, score) in enumerate(scored_boxes)]
    decorated_boxes.sort(key=lambda x: x[1])

    counter = 1
    pick = []
    while decorated_boxes:
        i = decorated_boxes[-1][2]
        pick.append(i)
        counter += 1

        a = decorated_boxes[-1]
        before = len(decorated_boxes)
        decorated_boxes = [box
                           for box in decorated_boxes[:-1]
                           if compute_overlap(a, box) <= max_overlap]

    return pick

def compute_overlap(a, b):
    area_a = (a[0][2]-a[0][0]+1) * (a[0][3]-a[0][1]+1)
    area_b = (b[0][2]-b[0][0]+1) * (b[0][3]-b[0][1]+1)

    x_overlap = max(0, min(a[0][2], b[0][2]) - max(a[0][0], b[0][0])) + 1
    y_overlap = max(0, min(a[0][3], b[0][3]) - max(a[0][1], b[0][1])) + 1
    intersect = x_overlap * y_overlap
    return intersect / (area_a + area_b - intersect)