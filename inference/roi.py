"""Two-stage ROI procedure (Sec. 3.2.2): threshold first-pass attention over
the four views, take the top-scoring connected regions, and crop them for
the second pass. No training involved."""
import numpy as np


def attention_to_rois(att_map, thresh_frac=0.6, max_rois=4, min_area=64):
    """att_map: 2D first-pass attention. Returns bounding boxes
    (y0, y1, x0, x1) of connected regions above thresh_frac * max."""
    from scipy import ndimage
    m = att_map >= thresh_frac * att_map.max()
    lab, n = ndimage.label(m)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < min_area:
            continue
        score = att_map[lab == i].sum()
        boxes.append((score, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    boxes.sort(reverse=True)
    return [b[1:] for b in boxes[:max_rois]]


def crop(image, box, out_size=512):
    """Crop box from a full view and resize to the second-pass budget."""
    import cv2
    y0, y1, x0, x1 = box
    return cv2.resize(image[y0:y1, x0:x1], (out_size, out_size),
                      interpolation=cv2.INTER_AREA)
