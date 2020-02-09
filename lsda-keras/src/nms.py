def non_max_suppression(boxes, overlap):
    """
    Greedily select high-scoring detections and skip detections that are significantly covered by
    a previously selected detection.

    NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    but an inner loop has been eliminated to significantly speed it
    up in the case of a large number of boxes

    Args:
        boxes (list of Box): 
        max_overlap (number): The maximum allowed overlap
    Returns:
        -
    """
    if not boxes:
        return []

    areas = [(box[2]-box[0]+1) * (box[3]-box[1]+1) for box in boxes]

    # Sort by y2, record 
    decorated_boxes = [(box, i) for i, box in boxes]
    decorated_boxes.sort(key=lambda x: x[0][3])

    pick = [0]*len(boxes)
    counter = 1
    while decorated_boxes:
        i = decorated_boxes[-1][1]
        pick[counter] = i
        counter += 1

        x1 = max(boxes[i][0], max(box[0] for box, _ in decorated_boxes))
        y1 = max(boxes[i][1], max(box[1] for box, _ in decorated_boxes))
        x2 = max(boxes[i][2], max(box[2] for box, _ in decorated_boxes))
        y2 = max(boxes[i][3], max(box[3] for box, _ in decorated_boxes))
        width, height = max(0, x2-x1+1), max(0, y2-y1+1)

        # Compute the amount of overlap between this box and the others
        intersect = width * height
        overlaps = [intersect / (areas[i] + area - intersect)
                    for area in areas[[idx for _, idx in decorated_boxes]]]

        # Filter out any boxes which overlap too much
        decorated_boxes = [(box, i)
                           for (box, i), overlap in zip(decorated_boxes, overlaps)
                           if overlap <= max_overlap]
    
    return pick[:counter]