import os
import sys
import subprocess
import cv2
import time

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
modnet_script = os.path.join(BASE_DIR, "MODNet", "run_onnx.py")
harmonizer_script = os.path.join(BASE_DIR, "Harmonizer", "try.py")
image_path = os.path.join(
    BASE_DIR,
    "Harmonizer",
    "demo",
    "image_harmonization",
    "example",
    "harmonized_deep_only.tiff"
)

print("===================================")
print(f"BASE_DIR          : {BASE_DIR}")
print(f"MODNet Script     : {modnet_script}")
print(f"Harmonizer Script : {harmonizer_script}")
print(f"Result Image Path : {image_path}")
print("===================================\n")

# --- RUN FIRST SCRIPT (MODNet) ---
result = subprocess.run([sys.executable, modnet_script])

if result.returncode == 0:
    print("\n✅ run_onnx.py finished successfully. Running try.py...\n")
    subprocess.run([sys.executable, harmonizer_script])
else:
    print("\n❌ run_onnx.py failed. try.py will not run.\n")
    sys.exit(1)

# --- DISPLAY FINAL IMAGE ---
if os.path.exists(image_path):
    img = cv2.imread(image_path)

    if img is not None:
        cv2.imshow("Final Result", img)
        print("🖼️ Displaying final image... (Press any key or wait 5 seconds)")
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print(f"⚠️ Could not read image file: {image_path}")
else:
    print(f"❌ Path not found: {image_path}")
