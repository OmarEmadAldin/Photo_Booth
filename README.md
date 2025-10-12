# 🖼️ Photo Booth – Portrait Matting & Harmonization

This project integrates two deep learning models — **MODNet** and **Harmonizer** — to create a seamless photo booth experience.

- **MODNet** handles **portrait matting** (background removal).
- **Harmonizer** adjusts **color and lighting consistency** between the subject and the background.
- Together, they produce realistic and visually consistent results for virtual backgrounds.

---

## 🧠 Project Overview

This repository combines:
- [MODNet (by ZHKKKe)](https://github.com/ZHKKKe/MODNet) for real-time background matting.
- [Harmonizer (by ZHKKKe)](https://github.com/ZHKKKe/Harmonizer) for image harmonization.
- **Custom integration and improvements** that allow both models to run in sequence on live or pre-captured images.

Key custom changes include:
- 🧩 Simplified model loading for both MODNet and Harmonizer.  
- ⚙️ Added GPU/CPU compatibility handling.  
- 🎨 Improved blending and lighting adjustments for more realistic compositing.  
- 📷 Real-time webcam support for photo booth applications.

---

## 🧱 Setup Instructions

### 1️ Clone This Repository

```bash

mkdir external
cd external

git clone https://github.com/ZHKKKe/MODNet.git
git clone https://github.com/ZHKKKe/Harmonizer.git

cd ..
```
### 2 Add the changed files 
```bash
https://github.com/OmarEmadAldin/Photo_Booth.git
```
add each one in each coressponding file


- **The changes for modnet**
  -- Background subtraction package code in which i use the pretrained model to made up a code that work on CPU and GPU
  -- Background Choosing is a simple ui for choosing which background needed to be in the output using mouse cursor and it's also a package
  -- run_prod.py the python code in which i call both packages and capture the image and save both image the composite and the mask in capture_result file in the harmonizer file
 **The changes for harmonizer**
  -- The capture_result file that receive from the code in modnet the two images
  -- try.py the harmonizer code that works both on gpu and cpu

## Final code
- run_both.py code it runs both codes using subprocess and runs one after the second one ends


