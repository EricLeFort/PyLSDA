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

    pick = [0]*len(scored_boxes)
    counter = 1
    while decorated_boxes:
        i = decorated_boxes[-1][2]
        pick[counter] = i
        counter += 1

        x1 = max(scored_boxes[i][0][0], max(box[0] for box, _, _ in decorated_boxes))
        y1 = max(scored_boxes[i][0][1], max(box[1] for box, _, _ in decorated_boxes))
        x2 = max(scored_boxes[i][0][2], max(box[2] for box, _, _ in decorated_boxes))
        y2 = max(scored_boxes[i][0][3], max(box[3] for box, _, _ in decorated_boxes))
        width, height = max(0, x2-x1+1), max(0, y2-y1+1)

        # Compute the amount of overlap between this box and the others
        intersect = width * height
        overlaps = [intersect / (areas[i] + area - intersect)
                    for area in [areas[idx] for _, _, idx in decorated_boxes]]

        # Filter out any boxes which overlap too much
        decorated_boxes = [(box, score, i)
                           for (box, score, i), overlap in zip(decorated_boxes[:-1], overlaps[:-1])
                           if overlap <= max_overlap]

    return pick[:counter]