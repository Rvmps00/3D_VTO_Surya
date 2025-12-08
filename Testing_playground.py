import cv2
import numpy as np
import glob

# --- CONFIGURATION ---
IMAGE_FOLDER = "calibration_images/*.png"

# YOUR TUNED VALUES
BLUR_SIZE = 15
THRESH_VAL = 223
INVERT_VAL = 1
MERGE_VAL = 0

# PHYSICAL MEASUREMENT
BLOCK_SPACING_MM = 40.0 

# --- STRICT LANDSCAPE CONFIGURATION ---
# We force the computer to look for 4 Columns (Wide) and 3 Rows (Tall)
GRID_ROWS = 3  
GRID_COLS = 4  
EXPECTED_TOTAL = 12

def get_obj_points():
    """
    Creates the 'Real World' coordinates for a 4x3 LANDSCAPE grid.
    """
    objp = np.zeros((EXPECTED_TOTAL, 3), np.float32)
    # Generate grid (Rows, Cols)
    mgrid = np.mgrid[0:GRID_ROWS, 0:GRID_COLS].T.reshape(-1, 2)
    # Flip to match X/Y
    objp[:, :2] = mgrid[:, ::-1] * BLOCK_SPACING_MM
    return objp

def robust_sort_grid(points):
    """
    Sorts points into a strict 4-wide, 3-tall grid.
    """
    points = np.array(points)
    
    # 1. Sort by Y (Rows)
    points = sorted(points, key=lambda p: p[1])
    
    sorted_points = []
    for r in range(GRID_ROWS):
        # Take the next 4 points (One Row)
        row_points = points[r*GRID_COLS : (r+1)*GRID_COLS]
        # Sort this row by X (Left to Right)
        row_points = sorted(row_points, key=lambda p: p[0])
        sorted_points.extend(row_points)
        
    return np.array(sorted_points, dtype=np.float32)

def main():
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found.")
        return

    print("--- STRICT CALIBRATION (LANDSCAPE ONLY) ---")
    print("REQUIREMENT: Hold board WIDE (4 blocks across, 3 blocks down)")
    print(" [y] YES - Keep image")
    print(" [n] NO  - Discard image")
    print(" [q] QUIT")
    
    obj_points_list = [] 
    img_points_list = [] 
    base_objp = get_obj_points()
    
    valid_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- DETECTION ---
        k = BLUR_SIZE if BLUR_SIZE % 2 == 1 else BLUR_SIZE + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        mode = cv2.THRESH_BINARY_INV if INVERT_VAL == 1 else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, THRESH_VAL, 255, mode)
        
        if MERGE_VAL > 0:
            k_m = MERGE_VAL * 2 + 1
            kernel = np.ones((k_m, k_m), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500 and area < 50000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append([cx, cy])

        if len(centers) == EXPECTED_TOTAL:
            try:
                # --- SORT FOR 4x3 LANDSCAPE ---
                sorted_pts = robust_sort_grid(centers)
                
                # --- VISUAL VERIFICATION ---
                display = img.copy()
                for i, pt in enumerate(sorted_pts):
                    # Color Gradient Blue->Red
                    color = (255-i*20, 0, i*20)
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 5, color, -1)
                    
                    # Draw Numbers
                    cv2.putText(display, str(i), (int(pt[0]), int(pt[1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw Grid Helper Text
                cv2.putText(display, "Ensure: 0=TopLeft, 3=TopRight", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Verify Strict Landscape", display)
                key = cv2.waitKey(0)
                
                if key == ord('y'):
                    print(f"[KEPT] {fname}")
                    obj_points_list.append(base_objp)
                    img_points_list.append(sorted_pts.astype(np.float32))
                    valid_count += 1
                elif key == ord('q'):
                    break
                else:
                    print(f"[DISCARDED] {fname}")

            except Exception as e:
                print(f"Sort Error {fname}: {e}")
        else:
            print(f"[Skip] {fname}: Found {len(centers)} blocks.")

    cv2.destroyAllWindows()

    if valid_count < 5:
        print(f"\nNot enough images ({valid_count}). Need 5+.")
        return

    print(f"\n--- Running Math on {valid_count} images ---")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, gray.shape[::-1], None, None
    )

    print("\n========== FINAL RESULT ==========")
    print(f"Re-projection Error: {ret:.4f}")
    print("Intrinsic Matrix:")
    print(mtx)
    print("Distortion:")
    print(dist)
    
    np.savez("astra_calibration_final.npz", mtx=mtx, dist=dist)

if __name__ == "__main__":
    main()