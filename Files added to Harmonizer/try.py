import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf

from src import model  # make sure your 'src/model.py' defines Harmonizer


# ----------------------------- Harmonizer Model Function on GPU -----------------------------
def harmonize_image_gpu(composite_path, mask_path, pretrained='', save_path='harmonized.png'):
    """
    Harmonize a single composite image using a deep learning model only (no classical blending).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Check paths ---
    if not os.path.exists(composite_path):
        raise FileNotFoundError(f"Composite image not found: {composite_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    if not os.path.exists(pretrained):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained}")

    # --- Load images ---
    comp = Image.open(composite_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # --- Load model ---
    print("Loading Harmonizer model...")
    harmonizer = model.Harmonizer().to(device)
    harmonizer.load_state_dict(torch.load(pretrained, map_location=device), strict=True)
    harmonizer.eval()

    # --- Prepare tensors ---
    comp_tensor = tf.to_tensor(comp).unsqueeze(0).to(device)
    mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)

    # --- Run harmonization ---
    print("Running harmonization (deep model only)...")
    with torch.no_grad():
        args = harmonizer.predict_arguments(comp_tensor, mask_tensor)
        harmonized = harmonizer.restore_image(comp_tensor, mask_tensor, args)[-1]

    # --- Convert tensor to image ---
    harmonized_np = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))
    print("Harmonization complete (deep model output).")

    # --- Save result ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    harmonized_img.save(save_path)
    print(f"Deep harmonized image saved to: {save_path}")

    return harmonized_img

# ----------------------------- Harmonizer Model Function on CPU -----------------------------

def harmonize_image_cpu(composite_path, mask_path, pretrained='', save_path='harmonized_cpu.png'):
    """
    Harmonize a single composite image using a deep learning model on CPU.
    """
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Check paths ---
    if not os.path.exists(composite_path):
        raise FileNotFoundError(f"Composite image not found: {composite_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    if not os.path.exists(pretrained):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained}")

    # --- Load images ---
    comp = Image.open(composite_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # --- Load model ---
    print("Loading Harmonizer model (CPU mode)...")
    harmonizer = model.Harmonizer().to(device)
    state_dict = torch.load(pretrained, map_location=device)
    harmonizer.load_state_dict(state_dict, strict=True)
    harmonizer.eval()

    # --- Prepare tensors ---
    comp_tensor = tf.to_tensor(comp).unsqueeze(0).to(device)
    mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)

    # --- Run harmonization ---
    print("Running harmonization (CPU, deep model only)...")
    with torch.no_grad():
        args = harmonizer.predict_arguments(comp_tensor, mask_tensor)
        harmonized = harmonizer.restore_image(comp_tensor, mask_tensor, args)[-1]

    # --- Convert tensor to image ---
    harmonized_np = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))
    print("Harmonization complete (CPU deep model output).")

    # --- Save result ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    harmonized_img.save(save_path)
    print(f"Deep harmonized image saved to: {save_path}")

    return harmonized_img


# ----------------------------- Example Usage -----------------------------
if __name__ == "__main__":
    composite_image = r"F:\Omar 3amora\Photo Booth\Harmonizer\capture_result\composite.jpg"
    mask_image = r"F:\Omar 3amora\Photo Booth\Harmonizer\capture_result\mask.jpg"
    pretrained_model = r"F:\Omar 3amora\Photo Booth\Harmonizer\pretrained\harmonizer.pth"

    output_path = r"F:\Omar 3amora\Photo Booth\Harmonizer\demo\image_harmonization\example\harmonized_deep_only.tiff"
    harmonize_image_cpu(composite_image, mask_image, pretrained_model, output_path)
