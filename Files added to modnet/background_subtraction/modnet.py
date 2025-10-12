import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.models.modnet import MODNet

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class ModNetWebCamGPU:
    def __init__(self, ckpt_path, cam_index=0, width=1280, height=720):
        self.device = get_device()
        print(f"Using device: {self.device.upper()}")

        self.cam_index = cam_index
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        # Load MODNet
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Camera
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ----------------------------- Apply MODNet -----------------------------
    def get_foreground(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        frame_resized = cv2.resize(frame_rgb, (672,512), cv2.INTER_AREA)
        frame_PIL = Image.fromarray(frame_resized)
        frame_tensor = self.transforms(frame_PIL).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, matte_tensor = self.model(frame_tensor, True)

        matte_np = matte_tensor[0,0].cpu().numpy()
        matte_np = cv2.resize(matte_np, (w,h))
        matte_np = np.expand_dims(matte_np,2)
        matte_np = np.repeat(matte_np,3,axis=2)
        return matte_np  # return matte only


class ModNetWebCamCPU:
    def __init__(self, ckpt_path, cam_index=0, width=1280, height=720):
        self.device = torch.device("cpu")
        print("Using device: CPU")

        # ---------------- Camera Setup ----------------
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # ---------------- Image Transform ----------------
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # ---------------- Load MODNet ----------------
        print("Loading pre-trained MODNet...")
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("MODNet ready on CPU.")

    # ---------------- Apply MODNet ----------------
    def get_foreground(self, frame):
        """Extract a clean matte mask for the given frame (CPU version, aligned with GPU)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Resize to MODNet expected input size (no crop)
        frame_resized = cv2.resize(frame_rgb, (672, 512), cv2.INTER_AREA)
        frame_PIL = Image.fromarray(frame_resized)

        # Convert to tensor
        frame_tensor = self.transforms(frame_PIL).unsqueeze(0)

        # Inference (CPU)
        with torch.no_grad():
            _, _, matte_tensor = self.model(frame_tensor, True)

        # Convert to numpy matte
        matte_np = matte_tensor[0, 0].cpu().numpy()
        matte_np = cv2.resize(matte_np, (w, h))
        matte_np = np.clip(matte_np, 0, 1)

        # Post-processing (mild edge refinement)
        matte_np = cv2.GaussianBlur(matte_np, (3, 3), 0)
        matte_np = np.power(matte_np, 1.2)  # improve contrast slightly

        # Make 3-channel matte for blending
        matte_np = np.repeat(matte_np[..., np.newaxis], 3, axis=2)

        return matte_np


    # ---------------- Matte Cleaner ----------------
    def clean_matte(self, matte):
        """Reduce halos and smooth edges."""
        # Increase contrast — make edges sharper
        matte = np.clip((matte - 0.25) * 1.6, 0, 1)

        # Morphological operations for refining mask
        kernel = np.ones((3, 3), np.uint8)
        matte = cv2.erode(matte, kernel, iterations=1)
        matte = cv2.GaussianBlur(matte, (5, 5), 0)

        return matte