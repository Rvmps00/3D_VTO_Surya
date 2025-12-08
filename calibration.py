import cv2
import numpy as np
import glob
import os

# --- YOU MUST CONFIGURE THESE ---
# 3x3 cm = 30mm
CHESSBOARD_SQUARE_SIZE_MM = 30 
# Your 9x6 board has 8x5 inner corners
CHESSBOARD_CORNERS = (8, 5)   
COLOR_CAMERA_INDEX = 0 
# ------------------------------

# Setup
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
objp = objp * CHESSBOARD_SQUARE_SIZE_MM

# Arrays to store points
obj_points = []  # 3D points in real world space
img_points = [] # 2D points in color image plane

print("Starting COLOR-ONLY calibration script...")
print(f"Board size: {CHESSBOARD_CORNERS}, Square size: {CHESSBOARD_SQUARE_SIZE_MM}mm")

color_cam = None

try:
    color_cam = cv2.VideoCapture(COLOR_CAMERA_INDEX)
    if not color_cam.isOpened():
        raise RuntimeError("Could not open color camera")

    # --- Set Camera Resolutions to Match Specs ---
    color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    w_color = color_cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_color = color_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Color Camera Initialized: {w_color}x{h_color}")

    print("\n--- INSTRUCTIONS ---")
    print("Show the (8, 5) board to the camera.")
    print("Press 'c' to capture a valid frame (when 'COLOR: OK' appears).")
    print("Press 'q' to finish and calibrate (15-20 captures recommended).")
    
    img_count = 0

    while True:
        ret, color_image_bgr = color_cam.read() # This is 1280x720
        if not ret:
            print("Failed to get color frame")
            continue
            
        gray_color = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- Find corners in Color Image (1280x720) ---
        ret_color, corners_color = cv2.findChessboardCorners(gray_color, CHESSBOARD_CORNERS, None)

        # --- Visualization ---
        cv2.drawChessboardCorners(color_image_bgr, CHESSBOARD_CORNERS, corners_color, ret_color)
        
        # --- Real-time Indicators ---
        if ret_color:
            cv2.putText(color_image_bgr, "COLOR: OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(color_image_bgr, "COLOR: NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # --------------------------------

        cv2.imshow("Color Camera (1280x720)", color_image_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if img_count < 15:
                print(f"Warning: You only have {img_count} images. 15-20+ is recommended.")
            break
        
        if key == ord('c'):
            if ret_color:
                print(f"Capture {img_count+1}... SUCCESS!")
                obj_points.append(objp)
                corners_color_subpix = cv2.cornerSubPix(gray_color, corners_color, (11, 11), (-1, -1), criteria)
                img_points.append(corners_color_subpix)
                img_count += 1
            else:
                print("Capture FAILED. 'COLOR: NOT FOUND'")

    print(f"\nCaptured {img_count} valid pairs. Starting calibration...")
    cv2.destroyAllWindows()
    
    h_color, w_color = gray_color.shape
    
    # --- Calibrate Color Camera ---
    print(f"Calibrating Color Camera ({w_color}x{h_color})...")
    ret_rgb, K_rgb, D_rgb, rvecs_rgb, tvecs_rgb = cv2.calibrateCamera(
        obj_points, img_points, (w_color, h_color), None, None
    )

    print("Calibration complete!")
    print(f"Color Camera Error (RMS): {ret_rgb}")
    print("Color Camera Matrix (K_rgb):\n", K_rgb)
    print("Color Distortion Coeffs (D_rgb):\n", D_rgb)

    print("Saving calibration data to 'color_calibration.npz'...")
    np.savez("color_calibration.npz", 
             K_rgb=K_rgb,
             D_rgb=D_rgb
             )
    print("Done! We have successfully calibrated the color camera.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if color_cam:
        color_cam.release()
    cv2.destroyAllWindows()