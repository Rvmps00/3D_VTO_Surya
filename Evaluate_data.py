import cv2
import numpy as np
import glob
import os

# --- 1. CONFIGURATION (UPDATED WITH YOUR NEW VALUES) ---
IMAGE_FOLDER = "calibration_images/*.png"

# YOUR NEW MAGIC NUMBERS
BLUR_SIZE = 3          # Smudges the checkerboard pattern
THRESH_VAL = 230       # Isolates blocks from the wall
INVERT_VAL = 1         # 1 = Invert (White blocks on Black background)
MERGE_VAL = 2          # "Digital Glue" to fix shattered blocks

# --- IMPORTANT: UPDATE THIS MEASUREMENT ---
# Distance from the center of one block to the center of the next block (in mm)
# Example: If block is 20mm and gap is 20mm, center-to-center is 40mm.
BLOCK_SPACING_MM = 40.0 

# GRID SETUP
GRID_ROWS = 4
GRID_COLS = 3
EXPECTED_TOTAL = 12

def get_obj_points():
    """
    Creates the 'Real World' coordinates of your blocks.
    Assume Z=0 (Flat Board).
    """
    objp = np.zeros((EXPECTED_TOTAL, 3), np.float32)
    mgrid = np.mgrid[0:GRID_ROWS, 0:GRID_COLS].T.reshape(-1, 2)
    
    # We flip columns/rows to match X/Y image coordinates
    # X = Column Index * Spacing
    # Y = Row Index * Spacing
    objp[:, :2] = mgrid[:, ::-1] * BLOCK_SPACING_MM
    return objp

def sort_grid_points(points, rows, cols):
    """
    Sorts random blob centers into a strict Top-Left to Bottom-Right order.
    """
    # 1. Sort by Y (Rows)
    points = sorted(points, key=lambda p: p[1])
    
    sorted_points = []
    for r in range(rows):
        row_points = points[r*cols : (r+1)*cols]
        # 2. Sort this row by X (Left to Right)
        row_points = sorted(row_points, key=lambda p: p[0])
        sorted_points.extend(row_points)
        
    return np.array(sorted_points, dtype=np.float32)

def main():
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found! Check your folder path.")
        return

    print(f"--- Starting Calibration on {len(images)} images ---")
    
    obj_points = [] # 3D points in real world space
    img_points = [] # 2D points in image plane
    
    base_objp = get_obj_points()
    valid_images = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- PROCESS IMAGE (YOUR TUNED PIPELINE) ---
        # 1. Blur
        k = BLUR_SIZE if BLUR_SIZE % 2 == 1 else BLUR_SIZE + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 2. Threshold
        mode = cv2.THRESH_BINARY_INV if INVERT_VAL == 1 else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, THRESH_VAL, 255, mode)

        # 3. Merge (Dilation)
        if MERGE_VAL > 0:
            k_m = MERGE_VAL * 2 + 1
            kernel = np.ones((k_m, k_m), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (Adjust these 500/50000 limits if needed)
            if area > 500 and area < 50000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append([cx, cy])

        # --- VALIDATE & STORE ---
        if len(centers) == EXPECTED_TOTAL:
            try:
                sorted_centers = sort_grid_points(centers, GRID_ROWS, GRID_COLS)
                obj_points.append(base_objp)
                img_points.append(sorted_centers)
                valid_images += 1
                print(f"[OK] {fname}")
            except Exception as e:
                print(f"[Sort Error] {fname}: {e}")
        else:
            print(f"[Skip] {fname}: Found {len(centers)} blocks.")

    print(f"\n--- Running Math on {valid_images} valid images... ---")
    
    if valid_images < 5:
        print("ERROR: Not enough valid images! Need at least 5.")
        return

    # --- THE CALIBRATION ---
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print("\n========== CALIBRATION RESULT ==========")
    print(f"Re-projection Error: {ret:.4f} pixels")
    print("(Target: < 1.0 is Good. < 0.5 is Excellent)")
    
    print("\nIntrinsic Matrix (Camera DNA):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    
    np.savez("astra_calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nSaved to 'astra_calibration_data.npz'!")

if __name__ == "__main__":
    main()