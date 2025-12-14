import cv2
import numpy as np
from orbbec_camera import OrbbecCamera

# --- CONFIGURATION ---
RGB_INDEX = 0 
FX = 360.0  
FY = 360.0  
DEPTH_SCALE = 5.15 

# FLIP SETTINGS (Keep what worked for you!)
FLIP_RGB = True
FLIP_DEPTH = False

def main():
    try:
        depth_cam = OrbbecCamera(use_ir=False, enable_registration=True)
    except Exception as e:
        print(f"Depth Camera Error: {e}")
        return

    rgb_cam = cv2.VideoCapture(RGB_INDEX)
    # FORCE RGB TO 640x480 (Matches Depth)
    rgb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    rgb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not rgb_cam.isOpened():
        print(f"Error: RGB Camera {RGB_INDEX} not found")
        return

    print("--- MANUAL ALIGNMENT TOOL ---")
    print(" [W/A/S/D] Move Depth Overlay")
    print(" [Q] Quit and Print Values")
    
    # Starting Offsets
    shift_x = 0
    shift_y = 0

    while True:
        depth_data = depth_cam.get_frame() 
        ret, rgb_frame = rgb_cam.read()
        if not ret: break

        # 1. Apply Flips
        if FLIP_RGB: rgb_frame = cv2.flip(rgb_frame, 1)
        if FLIP_DEPTH: depth_data = cv2.flip(depth_data, 1)

        # 2. Prepare Depth Color Map
        depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # 3. Resize Depth to match RGB (Crucial)
        h, w, _ = rgb_frame.shape
        depth_color = cv2.resize(depth_color, (w, h))

        # 4. APPLY SHIFT (Translation Matrix)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        depth_shifted = cv2.warpAffine(depth_color, M, (w, h))

        # 5. Blend
        blended = cv2.addWeighted(rgb_frame, 0.7, depth_shifted, 0.4, 0)

        # 6. UI Text
        cv2.putText(blended, f"Shift X: {shift_x} | Shift Y: {shift_y}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(blended, (w//2-2, h//2-20), (w//2+2, h//2+20), (0,0,255), -1)
        cv2.rectangle(blended, (w//2-20, h//2-2), (w//2+20, h//2+2), (0,0,255), -1)

        cv2.imshow("Align Tool", blended)
        
        # 7. Keyboard Control
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'): shift_y -= 1  # Up
        elif key == ord('s'): shift_y += 1  # Down
        elif key == ord('a'): shift_x -= 1  # Left
        elif key == ord('d'): shift_x += 1  # Right
        elif key == ord('r'): shift_x, shift_y = 0, 0 # Reset

    print("\n" + "="*30)
    print(f"FINAL ALIGNMENT VALUES:")
    print(f"SHIFT_X = {shift_x}")
    print(f"SHIFT_Y = {shift_y}")
    print("="*30)

    depth_cam.close()
    rgb_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()