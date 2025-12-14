import cv2
import numpy as np
import glob
import os

def main():
    # Matches the file saved by manual_calib_tool_v3.py
    CALIB_FILE = "astra_calibration_manual.npz" 
    IMAGE_FOLDER = "calibration_images/*.png"
    
    if not os.path.exists(CALIB_FILE):
        print(f"File '{CALIB_FILE}' not found!")
        return

    # 1. Load the Data
    with np.load(CALIB_FILE) as X:
        mtx = X['mtx']
        # --- THE FIX ---
        # We IGNORE the calculated distortion (which causes the vortex).
        # We force it to zero.
        dist = np.zeros((1, 5)) 

    print("--- Testing Intrinsics ONLY (Ignoring Distortion) ---")
    print(f"Focal Length X (fx): {mtx[0,0]:.2f} (Should be ~500-600)")
    print(f"Focal Length Y (fy): {mtx[1,1]:.2f} (Should be ~500-600)")
    
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found.")
        return
    
    print("Press [Space] for next image. [q] to Quit.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 2. Get Optimal New Camera Matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 3. Undistort (With ZERO distortion)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # 4. Visualization
        combined = np.hstack((img, dst))
        scale = 0.8
        disp_w = int(combined.shape[1] * scale)
        disp_h = int(combined.shape[0] * scale)
        view = cv2.resize(combined, (disp_w, disp_h))
        
        cv2.putText(view, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(view, "Corrected (Matrix Only)", (disp_w//2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Sanity Check", view)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()