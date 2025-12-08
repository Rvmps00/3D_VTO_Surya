import cv2
import glob
import numpy as np

def nothing(x):
    pass

def main():
    # --- CONFIGURATION ---
    IMAGE_FOLDER = "calibration_images/*.png"
    
    images = glob.glob(IMAGE_FOLDER)
    if not images:
        print("No images found!")
        return
    
    # Pick the middle image (usually representative)
    target_img = cv2.imread(images[len(images)//2]) 
    if target_img is None:
        print("Could not load image.")
        return

    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    cv2.namedWindow("Tuner V2")
    
    # --- CONTROLS ---
    cv2.createTrackbar("Blur", "Tuner V2", 1, 15, nothing)     # <--- NEW! Smudges the checkerboard
    cv2.createTrackbar("Thresh", "Tuner V2", 211, 255, nothing) 
    cv2.createTrackbar("Invert", "Tuner V2", 1, 1, nothing)    
    cv2.createTrackbar("Merge", "Tuner V2", 0, 20, nothing)

    print("--- Controls ---")
    print(" 1. Increase 'Blur' to fuse the shattered white squares.")
    print(" 2. Tweak 'Thresh' to isolate the blocks.")
    print(" 3. Aim for exactly 12 Blocks.")

    while True:
        blur_val = cv2.getTrackbarPos("Blur", "Tuner V2")
        thresh_val = cv2.getTrackbarPos("Thresh", "Tuner V2")
        invert_val = cv2.getTrackbarPos("Invert", "Tuner V2")
        merge_val = cv2.getTrackbarPos("Merge", "Tuner V2")

        # 1. BLUR (The Fix) - Must be odd number
        k_size = blur_val * 2 + 1
        blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

        # 2. Threshold
        mode = cv2.THRESH_BINARY_INV if invert_val == 1 else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, thresh_val, 255, mode)

        # 3. Morphological Merge
        if merge_val > 0:
            kernel_size = merge_val * 2 + 1 
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        block_count = 0
        display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Sort contours by area to filter tiny noise
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (Adjust these bounds if needed)
            if area > 500 and area < 50000:
                block_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Draw Box
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw Center
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)

        # Status
        color = (0, 255, 0) if block_count == 12 else (0, 0, 255)
        cv2.putText(display_img, f"Blocks Found: {block_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_img, f"Blur: {k_size}x{k_size}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Tuner V2", display_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n[FINAL SETTINGS] Blur: {blur_val}, Thresh: {thresh_val}, Invert: {invert_val}, Merge: {merge_val}")

if __name__ == "__main__":
    main()