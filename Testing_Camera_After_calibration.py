import cv2
import numpy as np
import glob
import os

def main():
    # --- CONFIGURATION ---
    # Update this filename to match your saved calibration file
    CALIB_FILE = "astra_calibration_final.npz" 
    IMAGE_FOLDER = "calibration_images/*.png"
    
    if not os.path.exists(CALIB_FILE):
        print(f"File '{CALIB_FILE}' not found!")
        print("Did you run the calibration script successfully yet?")
        return

    # 1. Load the Calibration Data
    try:
        with np.load(CALIB_FILE) as X:
            mtx, dist = X['mtx'], X['dist']
            # Some scripts save 'rvecs'/'tvecs', some don't. We only need mtx/dist.
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("--- Loaded Camera Parameters ---")
    print(f"Focal Length X (fx): {mtx[0,0]:.2f}")
    print(f"Focal Length Y (fy): {mtx[1,1]:.2f}")
    print(f"Optical Center: ({mtx[0,2]:.2f}, {mtx[1,2]:.2f})")
    print("--------------------------------")
    
    # Teacher Note: 
    # For Astra Pro Plus (640x480), we expect fx and fy to be roughly 500-600.
    # If they are < 100 or > 2000, the calibration is wrong.

    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found to test with.")
        return
    
    print("Press [Space] to next. [q] to Quit.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # 2. Calculate New Camera Matrix (Optimized)
        # alpha=1 ensures we see the black borders (no cropping)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 3. Undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # 4. Visualization
        # Stack images side-by-side
        combined = np.hstack((img, dst))
        
        # Resize to fit screen
        scale = 0.8
        disp_w = int(combined.shape[1] * scale)
        disp_h = int(combined.shape[0] * scale)
        view = cv2.resize(combined, (disp_w, disp_h))
        
        cv2.putText(view, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(view, "Undistorted (Corrected)", (disp_w//2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Calibration Test", view)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()