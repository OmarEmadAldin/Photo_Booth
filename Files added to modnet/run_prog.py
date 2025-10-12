import cv2
import numpy as np
import os
import time
from datetime import datetime
from background_choosing.ui import UIManager, overlay_image
from background_subtraction.modnet import ModNetWebCamGPU , ModNetWebCamCPU


# =============================================================
#                      CONFIGURATION
# =============================================================
CKPT_PATH = r"F:\Omar 3amora\Photo Booth\MODNet\pretrained\modnet_webcam_portrait_matting.ckpt"
SAVE_DIR = r"F:\Omar 3amora\Photo Booth\Harmonizer\capture_result"
CAM_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
WINDOW_NAME = "Photo Booth"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================
#                    INITIALIZATION
# =============================================================
ui = UIManager(screen_width=FRAME_WIDTH, screen_height=FRAME_HEIGHT)
modnet = ModNetWebCamCPU(ckpt_path=CKPT_PATH, cam_index=CAM_INDEX)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, ui.mouse_callback)
print(" Photo Booth initialized successfully.")


# =============================================================
#                    HELPER FUNCTIONS
# =============================================================

def composite_foreground(frame, matte, bg_fit):
    """Blend foreground and background using the matte mask."""
    matte = np.expand_dims(matte, axis=2) if matte.ndim == 2 else matte
    matte = np.clip(matte, 0, 1).astype(np.float32)

    fg_only = matte * frame
    if bg_fit is not None:
        blended = fg_only + (1 - matte) * bg_fit
    else:
        blended = frame.copy()

    return np.uint8(blended)


def save_capture(blended, matte):
    """Save composite image and matte mask with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    composite_path = os.path.join(SAVE_DIR, f"composite_{timestamp}.tiff")
    mask_path = os.path.join(SAVE_DIR, f"mask_{timestamp}.jpg")

    # Save uncompressed TIFF for maximum quality
    cv2.imwrite(composite_path, blended, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    cv2.imwrite(mask_path, np.uint8(matte * 255))

    print(f" Saved composite → {composite_path}")
    print(f" Saved mask → {mask_path}")


# =============================================================
#                      MAIN LOOP
# =============================================================

try:
    print(" Starting live preview... Press 'q' to quit.")

    while True:
        ret, frame = modnet.cap.read()
        if not ret or frame is None:
            print(" Camera frame not available.")
            time.sleep(0.1)
            continue

        # Mirror effect for natural interaction
        frame = cv2.flip(frame, 1)

        # Get matte mask from MODNet
        matte = modnet.get_foreground(frame)  # float [0,1]

        # Create composited background
        blended = composite_foreground(frame, matte, ui.bg_fit)

        # Draw UI and handle countdown
        display_frame = blended.copy()
        ui.draw_buttons(display_frame)
        countdown_done = ui.handle_countdown(display_frame)

        # When countdown finishes → capture
        if countdown_done:
            save_capture(blended, matte)

        # Display preview
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(" Exiting Photo Booth...")
            break

except KeyboardInterrupt:
    print("\n Interrupted by user.")

except Exception as e:
    print(f" Unexpected error: {e}")

finally:
    # Graceful cleanup
    modnet.cap.release()
    cv2.destroyAllWindows()
    print(" Resources released cleanly.")
