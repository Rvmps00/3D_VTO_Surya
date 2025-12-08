# Smart Mirror VTO: Orbbec Astra Pro Plus Framework

This repository contains a complete framework for a **Real-Time Virtual Try-On (VTO)** system using the **Orbbec Astra Pro Plus** depth camera. 

It covers the full pipeline: from deep-dive driver modifications and custom **Intrinsic Camera Calibration** (addressing IR absorption issues) to the final **Augmented Reality application** that maps clothing onto a user using Computer Vision and MediaPipe.

## 📝 Project Overview

This system captures a video feed, detects the user's body landmarks in real-time, and warps clothing images (torso and sleeves) to fit the user's pose. It addresses specific hardware challenges associated with the Astra Pro Plus, including unlocking the hidden Infrared (IR) stream and calibrating using custom 3D-printed targets with discontinuous geometry.

### Key Features
* **Hybrid Camera Support:** Utilizes the Orbbec Astra Pro Plus for both RGB (Color) and IR/Depth streams.
* **Robust Calibration:** Includes a research-grade manual annotation tool to calibrate against "shattered" or low-contrast targets.
* **Articulated Warping:** Splits clothing into separate parts (torso, left sleeve, right sleeve) to allow for natural arm movement.
* **Jitter Stabilization:** Implements an **Exponential Moving Average (EMA)** filter to smooth out MediaPipe landmark flickering.
* **Anti-Tearing Logic:** Includes mathematical sanity checks to prevent texture artifacts when arms overlap the body.

---

## 🛠 Hardware Requirements

* **Camera:** Orbbec Astra Pro Plus.
* **Platform:** Linux (Tested on Ubuntu) or Windows.
* **Calibration Target:** Custom 3D Printed Board (Required for Research Calibration).
    * **Grid:** 4 Blocks Wide x 3 Blocks Tall (12 Blocks Total).
    * **Pattern:** Each block is a 2x2 checkerboard.
    * **Spacing:** 40.0mm Center-to-Center distance.

---

## ⚙️ Prerequisites & Installation

### 1. Python Dependencies
```bash
pip install opencv-python numpy mediapipe
````

### 2\. OpenNI2 SDK Setup

Ensure you have the OpenNI2 SDK installed. This project relies on `libOpenNI2.so` (Linux) or `OpenNI2.dll` (Windows).

**Important:** Update the `ORBBEC_LIB_PATH` variable in `orbbec_camera.py` to point to your SDK:

```python
ORBBEC_LIB_PATH = "/path/to/your/OpenNI_2.3.0.../sdk/libs/libOpenNI2.so"
```

### 3\. ⚠️ The Critical Driver Hack (Unlock IR)

By default, the Astra Pro Plus driver on Linux may hide the IR stream or misreport it as Depth. You must edit the driver configuration file to force "Uncompressed" mode for calibration to work.

1.  Navigate to your OpenNI2 Drivers folder (e.g., `.../OpenNI2/Drivers/`).
2.  Open `Orbbec.ini` (or `PS1080.ini`).
3.  Scroll to the `[IR]` section and **Uncomment/Edit** these lines:

<!-- end list -->

```ini
[IR]
; Remove the semicolon (;) to uncomment
OutputFormat=203      ; Sets format to 16-bit Grayscale
InputFormat=0         ; Forces Uncompressed mode (Critical!)
Mirror=1
Resolution=1          ; Forces VGA (640x480)
FPS=30
```

*Note: After saving this file, unplug and replug the camera.*

-----

## 🚀 Usage Phase 1: Calibration (Research)

Due to IR absorption properties of 3D printing materials (Black PLA), automated corner detection often fails (the "Shattered Block" effect). We use a manual annotation tool for 100% accuracy.

### 1\. Data Collection

Run the manual tool to annotate the 12 block centers.

```bash
python manual_calib_tool_v3.py
```

  * **Click order:** Left-to-Right, Row-by-Row.
  * **Keys:** `Space` (Save), `d` (Discard), `q` (Finish & Calibrate).

### 2\. Validation

Verify the calibration by undistorting the camera feed.

```bash
python test_undistortion.py
```

-----

## 🚀 Usage Phase 2: Virtual Try-On (Application)

Once the camera is calibrated and the environment is set up, run the main VTO application.

### 1\. Prepare Assets

Ensure your clothing images are transparent PNGs and placed in the correct directory (e.g., `Baju 2D/over-man/`).

  * `dark-denim-mid part.png` (Torso)
  * `dark-denim-arm.png` (Right Sleeve)
  * `dark-denim-arm-1.png` (Left Sleeve)

### 2\. Run the System

```bash
python main.py
```

### 3\. Controls & Configuration

  * **'q':** Quit the application.
  * **Smoothing:** Adjust `LandmarkSmoother(alpha=0.3)` in `main.py`.
      * Lower `alpha` (e.g., 0.1) = Smoother but more latency.
      * Higher `alpha` (e.g., 0.8) = Snappier but more jitter.

-----

## 🧠 Technical Challenges Solved

### Calibration Challenges

  * **The "Ghost" Depth Issue:** Fixed by forcing `ONI_SENSOR_IR` (ID 3) and hardcoding `InputFormat=0` in the driver `.ini`.
  * **The "Shattered Block" Effect:** Addressed via the Manual Annotation Tool (`manual_calib_tool_v3.py`) which ignores internal block noise.
  * **The "Impossible Matrix" (f=3600):** Fixed by implementing a strict Left-to-Right, Row-by-Row clicking protocol that ensures correct geometric mapping.

### VTO Challenges

  * **Jitter:** Solved using Exponential Moving Average (EMA) filtering on MediaPipe landmarks.
  * **Texture Tearing:** Solved by implementing geometric sanity checks to prevent "division by zero" errors when arms are orthogonal to the camera.

-----





