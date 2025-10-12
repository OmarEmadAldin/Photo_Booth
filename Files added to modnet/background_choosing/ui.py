import cv2
import numpy as np
import time

# ----------------------------- Helper functions -----------------------------
def safe_load_and_resize(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]

    # Clip overlay if outside frame
    if y + h > bh:
        h = bh - y
        overlay = overlay[:h, :, :]
    if x + w > bw:
        w = bw - x
        overlay = overlay[:, :w, :]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = alpha*overlay[:,:,c] + (1-alpha)*background[y:y+h, x:x+w, c]
    else:
        background[y:y+h, x:x+w] = overlay

# ----------------------------- UI Manager -----------------------------
class UIManager:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Background buttons
        button_paths = [
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\Pyramids.jpg',
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\KarnakTemple.jpg',
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\AbuSimbelTemples.jpg'
        ]
        bg_paths = [
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\Pyramids.png',
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\KarnakTemple.png',
            r'F:\Omar 3amora\Photo Booth\MODNet\Images\AbuSimbelTemples.png'
        ]
        self.button_imgs = [safe_load_and_resize(p, (200,200)) for p in button_paths]
        self.bg_imgs = [safe_load_and_resize(p, (screen_width, screen_height)) for p in bg_paths]

        self.BUTTON_SIZE = (200,200)
        self.positions = [(screen_width - 350, 50),
                          (screen_width - 350, 300),
                          (screen_width - 350, 550)]
        self.buttons = [(x, y, self.BUTTON_SIZE[0], self.BUTTON_SIZE[1]) for (x, y) in self.positions]

        # Shoot button
        self.shoot_w, self.shoot_h = 200, 80
        self.shoot_x = (screen_width - self.shoot_w)//2
        self.shoot_y = screen_height - self.shoot_h - 40
        self.shoot_button = (self.shoot_x, self.shoot_y, self.shoot_w, self.shoot_h)

        # Hover & countdown
        self.hover_index = -1
        self.GLOW_COLOR = (0, 255, 255)
        self.GLOW_THICKNESS = 5
        self.SHADOW_OFFSET = 8
        self.countdown_active = False
        self.countdown_start = 0
        self.COUNTDOWN_TIME = 3
        self.bg_fit = None

    # ----------------------------- Mouse Callback -----------------------------
    def mouse_callback(self, event, x, y, flags, param):
        self.hover_index = -1
        for i, (bx, by, bw, bh) in enumerate(self.buttons):
            if bx <= x <= bx+bw and by <= y <= by+bh:
                self.hover_index = i
        sx, sy, sw, sh = self.shoot_button
        if sx <= x <= sx+sw and sy <= y <= sy+sh:
            self.hover_index = 100

        # Click events
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (bx, by, bw, bh) in enumerate(self.buttons):
                if bx <= x <= bx+bw and by <= y <= by+bh:
                    self.bg_fit = self.bg_imgs[i]
            if sx <= x <= sx+sw and sy <= y <= sy+sh:
                self.start_countdown()

    # ----------------------------- Countdown -----------------------------
    def start_countdown(self):
        self.countdown_active = True
        self.countdown_start = time.time()

    def handle_countdown(self, frame):
        if not self.countdown_active:
            return False
        elapsed = time.time() - self.countdown_start
        remaining = self.COUNTDOWN_TIME - elapsed
        if remaining > 0:
            # Circular countdown
            center = (self.screen_width//2, self.screen_height//2)
            radius = 60
            angle = int(360*(remaining/self.COUNTDOWN_TIME))
            cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0,255,255), 10)
            cv2.putText(frame, str(int(remaining)+1),
                        (center[0]-30, center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 5)
            return False
        else:
            filename = f"captured_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            self.countdown_active = False
            return True

    # ----------------------------- Draw Buttons -----------------------------
    def draw_buttons(self, frame):
        for i, (img, (x, y)) in enumerate(zip(self.button_imgs, self.positions)):
            overlay_image(frame, np.full(img.shape, 50, dtype=np.uint8), x+self.SHADOW_OFFSET, y+self.SHADOW_OFFSET)
            overlay_image(frame, img, x, y)
            if self.hover_index == i:
                cv2.rectangle(frame, (x-5, y-5), (x+self.BUTTON_SIZE[0]+5, y+self.BUTTON_SIZE[1]+5),
                              self.GLOW_COLOR, self.GLOW_THICKNESS)
        # Shoot button
        sx, sy, sw, sh = self.shoot_button
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (50,50,50), -1, cv2.LINE_AA)
        cv2.putText(frame, "SHOOT", (sx+40, sy+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if self.hover_index == 100:
            cv2.rectangle(frame, (sx-5, sy-5), (sx+sw+5, sy+sh+5), self.GLOW_COLOR, self.GLOW_THICKNESS)
