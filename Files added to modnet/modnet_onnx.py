import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.models.modnet import MODNet
import onnxruntime as ort

class ModNetWebCamCPU_ONNX:
    
    def __init__(self, onnx_path, cam_index=0, width=1280, height=720):
        print("Using ONNX Runtime on CPU")

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

        # ---------------- Load ONNX Model ----------------
        print("Loading ONNX MODNet...")
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Debug: Print model input/output information
        print("Model inputs:")
        for input_meta in self.session.get_inputs():
            print(f"  {input_meta.name}: {input_meta.shape}")
        print("Model outputs:")
        for output_meta in self.session.get_outputs():
            print(f"  {output_meta.name}: {output_meta.shape}")
            
        print("ONNX MODNet ready on CPU.")

    # ---------------- Apply MODNet ----------------
    def get_foreground(self, frame):
        """Extract a clean matte mask for the given frame using ONNX model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Resize to MODNet expected input size (no crop)
        frame_resized = cv2.resize(frame_rgb, (672, 512), cv2.INTER_AREA)
        frame_PIL = Image.fromarray(frame_resized)

        # Convert to tensor
        frame_tensor = self.transforms(frame_PIL).unsqueeze(0)

        # Convert to numpy for ONNX inference
        input_numpy = frame_tensor.numpy()

        # ONNX inference - use the correct input name
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_numpy})
        
        # Debug: Print output information
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
        
        # Get the matte output - try different indices
        if len(outputs) >= 3:
            matte_tensor = torch.from_numpy(outputs[2])
        elif len(outputs) >= 2:
            matte_tensor = torch.from_numpy(outputs[1])
        else:
            matte_tensor = torch.from_numpy(outputs[0])
            
        matte_np = matte_tensor[0, 0].numpy()
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

