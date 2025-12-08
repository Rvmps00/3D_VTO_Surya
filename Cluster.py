import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
IMAGE_FOLDER = "calibration_images/*.png"
BLOCK_SPACING_MM = 40.0 

# GRID
GRID_ROWS = 3
GRID_COLS = 4
EXPECTED_POINTS = 12

# Global variables for mouse callback
clicks = []
current_img = None
display_img = None

def get_obj_points():
    objp = np.zeros((EXPECTED_POINTS, 3), np.float32)
    mgrid = np.mgrid[0:GRID_ROWS, 0:GRID_COLS].T.reshape(-1, 2)
    objp[:, :2] = mgrid[:, ::-1] * BLOCK_SPACING_MM
    return objp

def mouse_callback(event, x, y, flags, param):
    global clicks, current_img, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < EXPECTED_POINTS:
            clicks.append((x, y))
            # Draw visual feedback
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_img, str(len(clicks)-1), (x+5, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Manual Calibration", display_img)

def main():
    global clicks, current_img, display_img
    
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found.")
        return

    print("--- MANUAL CALIBRATION TOOL V2 (FIXED) ---")
    print(" 1. Click 12 centers (0=Top-Left -> 11=Bot-Right).")
    print(" 2. Press [Space] to save and go to next image.")
    print(" 3. Press [d] to Skip/Discard image.")
    print(" 4. Press [q] to FINISH and Calculate.")

    obj_points_list = []
    img_points_list = []
    base_objp = get_obj_points()
    
    valid_count = 0
    stop_program = False

    cv2.namedWindow("Manual Calibration")
    cv2.setMouseCallback("Manual Calibration", mouse_callback)

    for fname in images:
        if stop_program: break
        
        clicks = []
        current_img = cv2.imread(fname)
        if current_img is None: continue
        
        display_img = current_img.copy()
        cv2.putText(display_img, f"Img {valid_count+1}: Click 12 points", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Manual Calibration", display_img)
        
        while True:
            key = cv2.waitKey(1)
            
            # [Space] = Save & Next
            if key == ord(' '):
                if len(clicks) == EXPECTED_POINTS:
                    print(f"[SAVED] {fname}")
                    obj_points_list.append(base_objp)
                    img_points_list.append(np.array(clicks, dtype=np.float32))
                    valid_count += 1
                    break
                else:
                    print(f"Need 12 points! You have {len(clicks)}.")
            
            # [z] = Undo
            elif key == ord('z'):
                if len(clicks) > 0:
                    clicks.pop()
                    display_img = current_img.copy()
                    cv2.putText(display_img, f"Img {valid_count+1}: Click 12 points", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    for i, pt in enumerate(clicks):
                        cv2.circle(display_img, pt, 5, (0, 0, 255), -1)
                        cv2.putText(display_img, str(i), (pt[0]+5, pt[1]-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Manual Calibration", display_img)

            # [d] = Discard (Skip image)
            elif key == ord('d'):
                print(f"[SKIPPED] {fname}")
                break
                
            # [q] = Quit and Calculate
            elif key == ord('q'):
                print("Finishing up...")
                stop_program = True
                break

    cv2.destroyAllWindows()

    if valid_count < 5:
        print(f"Not enough images annotated ({valid_count}). Need at least 5.")
        return

    print(f"\n--- Running Math on {valid_count} Manual Images ---")
    gray_shape = current_img.shape[:2][::-1] 
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, gray_shape, None, None
    )

    print("\n========== FINAL RESULT ==========")
    print(f"Re-projection Error: {ret:.4f}")
    print("Intrinsic Matrix:")
    print(mtx)
    print("Distortion:")
    print(dist)
    
    np.savez("astra_calibration_manual.npz", mtx=mtx, dist=dist)
    print("Saved to 'astra_calibration_manual.npz'")

if __name__ == "__main__":
    main()