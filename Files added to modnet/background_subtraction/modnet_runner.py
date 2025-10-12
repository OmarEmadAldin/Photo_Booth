import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.models.modnet import MODNet

# -------------------------------
# Helper function for device
# -------------------------------
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------------
# Class 1: MODNet Image Processor
# -------------------------------
class ModNetImage:
    def __init__(self, ckpt_path):
        self.device = get_device()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def process(self, image_path, save_path=None, show=False):
        frame_np = cv2.imread(image_path)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
        frame_np = frame_np[:, 120:792, :]
        frame_np = cv2.flip(frame_np, 1)

        frame_PIL = Image.fromarray(frame_np)
        frame_tensor = self.transforms(frame_PIL).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, matte_tensor = self.model(frame_tensor, True)

        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte_np = matte_tensor[0].cpu().numpy().transpose(1, 2, 0)
        fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0, dtype=np.float32)
        result = np.uint8(fg_np)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if save_path:
            cv2.imwrite(save_path, result_bgr)
        if show:
            cv2.imshow("Result", result_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result_bgr


# -------------------------------
# Class 2: MODNet Video Processor
# -------------------------------
class ModNetVideo:
    def __init__(self, ckpt_path):
        self.device = get_device()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def process(self, input_path, output_path, save_matte=False, fps=60):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            new_h, new_w = h - (h % 32), w - (w % 32)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor = self.to_tensor(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, matte = self.model(tensor, True)

            matte = torch.nn.functional.interpolate(
                matte, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            matte = matte.repeat(1, 3, 1, 1)
            matte_np = matte[0].cpu().numpy().transpose(1, 2, 0)

            if save_matte:
                result = (matte_np * 255).astype(np.uint8)
            else:
                frame_rgb_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = (matte_np * frame_rgb_orig + (1 - matte_np) * 255).astype(np.uint8)

            out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()


# -------------------------------
# Class 3: MODNet WebCam Processor
# -------------------------------
class ModNetWebCam:
    def __init__(self, ckpt_path, cam_index, width=1280, height=720):
        # Auto-detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")

        self.cam_index = cam_index
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Load MODNet
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Initialize camera
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def start(self):
        while True:
            ret, frame_np = self.cap.read()
            if not ret:
                break

            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
            frame_np = frame_np[:, 120:792, :]
            frame_np = cv2.flip(frame_np, 1)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = self.transforms(frame_PIL).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, matte_tensor = self.model(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].cpu().numpy().transpose(1, 2, 0)
            fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
            view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

            cv2.imshow("MODNet - WebCam [Press 'Q' To Exit]", view_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
