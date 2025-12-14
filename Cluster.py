import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
IMAGE_FOLDER = "calibration_images/*.png"
BLOCK_SPACING_MM = 40.0 

# GRID (Landscape 4x3)
GRID_ROWS = 3
GRID_COLS = 4
EXPECTED_POINTS = 12                                                                                                     

# Global variables
clicks = []
current_img = None
display_img = None

def get_obj_points():
    """
    FIXED GENERATION:
    Creates points strictly Left-to-Right, Top-to-Bottom.
    (0,0), (40,0), (80,0), (120,0) -> Row 0
    (0,40), (40,40)...             -> Row 1
    """
    objp = []
    for r in range(GRID_ROWS):          # Iterate Rows (0, 1, 2)
        for c in range(GRID_COLS):      # Iterate Cols (0, 1, 2, 3)
            # X = Col * Spacing, Y = Row * Spacing, Z = 0
            objp.append([c * BLOCK_SPACING_MM, r * BLOCK_SPACING_MM, 0])
    return np.array(objp, dtype=np.float32)

def mouse_callback(event, x, y, flags, param):
    global clicks, current_img, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < EXPECTED_POINTS:
            clicks.append((x, y))
            # Visual Feedback
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            # Draw number to confirm order
            cv2.putText(display_img, str(len(clicks)-1), (x+5, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Manual Calibration V3", display_img)

def main():
    global clicks, current_img, display_img
    
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found.")
        return

    print("--- MANUAL CALIBRATION V3 (FIXED MAP) ---")
    print(" 1. Click Row 1 (Left->Right)")
    print(" 2. Click Row 2 (Left->Right)")
    print(" 3. Click Row 3 (Left->Right)")
    print(" 4. Press [Space] to Save.")
    print(" 5. Press [d] to Skip.")
    print(" 6. Press [q] to Finish.")

    obj_points_list = []
    img_points_list = []
    base_objp = get_obj_points()
    
    valid_count = 0
    stop_program = False

    cv2.namedWindow("Manual Calibration V3")
    cv2.setMouseCallback("Manual Calibration V3", mouse_callback)

    for fname in images:
        if stop_program: break
        
        clicks = []
        current_img = cv2.imread(fname)
        if current_img is None: continue
        
        display_img = current_img.copy()
        cv2.putText(display_img, f"Img {valid_count+1}: Click 12 points", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Manual Calibration V3", display_img)
        
        while True:
            key = cv2.waitKey(1)
            
            # [Space] Save
            if key == ord(' '):
                if len(clicks) == EXPECTED_POINTS:
                    print(f"[SAVED] {fname}")
                    obj_points_list.append(base_objp)
                    img_points_list.append(np.array(clicks, dtype=np.float32))
                    valid_count += 1
                    break
                else:
                    print(f"Need 12 points! (Have {len(clicks)})")
            
            # [z] Undo
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
                    cv2.imshow("Manual Calibration V3", display_img)

            # [d] Discard
            elif key == ord('d'):
                print(f"[SKIPPED] {fname}")
                break
                
            # [q] Calculate
            elif key == ord('q'):
                print("Calculating...")
                stop_program = True
                break

    cv2.destroyAllWindows()

    if valid_count < 5:
        print(f"Not enough images ({valid_count}). Need 5+.")
        return

    print(f"\n--- Running Math on {valid_count} Images ---")
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