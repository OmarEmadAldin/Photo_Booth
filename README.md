# **Photo Booth – Intelligent Portrait Matting & Harmonization**

A complete photo booth system powered by deep learning, integrating **[MODNet](https://github.com/ZHKKKe/MODNet)** and **[Harmonizer](https://github.com/ZHKKKe/Harmonizer)** for realistic human-background compositing.  

This project enables seamless **real-time portrait matting**, **background selection**, and **lighting harmonization**, creating photo outputs that look natural and professionally blended.  

---

## **Features**

 **MODNet Integration** – High-quality human matting (background removal) in real-time.  
 **Harmonizer Integration** – Color and lighting adjustment for realistic subject-background blending.  
 **CPU/GPU Support** – Runs efficiently on both CPU and CUDA-enabled GPUs.  
 **Interactive UI** – Mouse-based background selection interface.  
 **Automated Workflow** – End-to-end pipeline that handles capturing, matting, and harmonization seamlessly.  
 **Clean Modular Design** – Two independent packages: `background_subtraction` and `background_choosing`, integrated by a unified controller.  

---

##  **Project Overview**

This repository merges and extends two state-of-the-art projects:  
- **[MODNet (by ZHKKKe)](https://github.com/ZHKKKe/MODNet)** → for real-time portrait matting.  
- **[Harmonizer (by ZHKKKe)](https://github.com/ZHKKKe/Harmonizer)** → for image harmonization.  

Your custom implementation integrates both in a single workflow with the following improvements:

###  Custom Enhancements
- **Simplified Model Loading:** Both models automatically initialize on available hardware (GPU/CPU).  
- **Unified Execution Flow:** Harmonizer automatically receives MODNet’s composite and mask outputs.  
- **Dynamic UI System:** Backgrounds can be chosen through a lightweight OpenCV-based user interface.  
- **Automation & Modularity:** Each stage (capture → matte → harmonize) runs independently or together.  
- **Improved Performance:** Optimized pre/post-processing for stable real-time webcam inference.  

---

##  **Repository Structure**

```
Photo_Booth/
│
├── background_subtraction/
│   ├── modnet_integration.py        # MODNet inference logic (CPU/GPU)
│   ├── modnet_webcam.py             # Real-time webcam capture & matting
│   └── __init__.py
│
├── background_choosing/
│   ├── ui_manager.py                # Simple UI for selecting background
│   ├── overlay_image.py             # Blending utility
│   └── __init__.py
│
├── external/
│   ├── MODNet/                      # Cloned original MODNet repo
│   └── Harmonizer/                  # Cloned original Harmonizer repo
│
├── modnet changes/                  # Your MODNet modifications
├── harmonizer changes/              # Your Harmonizer modifications
│
├── capture_results/                 # Auto-saved composites and masks
│
├── run_prod.py                      # Core runtime that captures and saves results
├── try.py                           # Modified Harmonizer main file (CPU/GPU compatible)
├── run_both.py                      # Orchestrates both pipelines via subprocess
│
└── README.md
```

---

##  **Setup Instructions**

###  Clone This Repository

```bash
git clone https://github.com/OmarEmadAldin/Photo_Booth.git
cd Photo_Booth
```

###  Clone the Original Repositories

```bash
mkdir external
cd external

git clone https://github.com/ZHKKKe/MODNet.git
git clone https://github.com/ZHKKKe/Harmonizer.git

cd ..
```

### 3️ Add Your Custom Changes

Copy your modified files into the corresponding projects:

```bash
# Apply MODNet modifications
cp -r "modnet changes/." "external/MODNet/"

# Apply Harmonizer modifications
cp -r "harmonizer changes/." "external/Harmonizer/"
```

---

##  **Dependencies**

Make sure you have **Python 3.8+** and the required dependencies:

```bash
pip install -r requirements.txt
```

If running on GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

##  **Usage**

You can use the project in three modes depending on your needs.

---

###  **1. Run MODNet Only (Background Subtraction)**

Run the portrait matting module that removes the background and saves both:
- the **composite image**
- the **alpha mask**

```bash
python run_prod.py
```

**Output:**  
- Saved inside `capture_results/`  
  - `composite.png`  
  - `mask.png`

This script:
- Loads the MODNet pretrained model (`modnet_webcam_portrait_matting.ckpt`)
- Captures the frame from the webcam
- Generates the matte and composite
- Saves both results for the next stage (Harmonizer)

---

###  **2. Run Harmonizer Only**

Harmonizes the images saved by MODNet.  

```bash
python harmonizer/try.py
```

This script:
- Loads both `composite.png` and `mask.png` from `capture_results/`
- Runs the Harmonizer model
- Produces the final harmonized image

**Output:**  
- `harmonized.png` in `external/Harmonizer/outputs/`

---

###  **3. Run the Full Integrated Pipeline**

To automatically run both MODNet and Harmonizer sequentially using `subprocess`:

```bash
python run_both.py
```

This script:
1. Launches MODNet for portrait matting  
2. Waits for the capture and result saving  
3. Runs Harmonizer on the results automatically  
4. Displays and saves the harmonized final output

---

##  **System Requirements**

| Component | Minimum | Recommended |
|------------|-----------|-------------|
| CPU | Intel i5 8th Gen | Intel i7 / AMD Ryzen 7 |
| GPU | None (CPU supported) | NVIDIA GPU with CUDA ≥ 11.8 |
| RAM | 8 GB | 16 GB |
| OS | Windows / Linux | Ubuntu 22.04 or Windows 11 |

---

##  **Pipeline Summary**

```text
[Webcam Frame]
     ↓
 [MODNet]
     ↓ produces → (Composite, Mask)
     ↓
 [Harmonizer]
     ↓
 [Final Harmonized Output]
```
 The integration ensures that the **subject is cleanly extracted** and **realistically blended** into any chosen background.

---

## **User Interface**

- A simple **OpenCV window** allows you to select a background interactively.
- Use the **mouse cursor** to choose the desired background image.
- The system overlays the subject on top of the selected background and harmonizes lighting automatically.


---
