def normalize_bboxes(bboxes, img_width=224, img_height=224):
    """Normalize bounding boxes to be in range [0,1]."""
    bboxes[:, [0, 2]] /= img_width   # Normalize x_min and x_max
    bboxes[:, [1, 3]] /= img_height  # Normalize y_min and y_max
    return bboxes
