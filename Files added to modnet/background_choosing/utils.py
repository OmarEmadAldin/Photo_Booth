import cv2
import numpy as np

def safe_load_and_resize(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
    else:
        background[y:y+h, x:x+w] = overlay
