import cv2
import numpy as np
from orbbec_camera import OrbbecCamera

def main():
    # --- CONFIGURATION ---
    # We now know False gives us the clear geometric view
    SHOW_IR_MODE = True 
    
    try:
        # Initialize Camera
        # We use use_ir=False because we want the Depth stream that actually works
        cam = OrbbecCamera(use_ir=False, enable_registration=True)
        
        print(" Controls:")
        print(" [s] Save Snapshot")
        print(" [q] Quit")
        
        frame_count = 0

        while True:
            # 1. Get Frame (Uint16 Depth Data)
            frame_data = cam.get_frame()

            # 2. Convert Depth to "Fake Image" for Calibration
            # We normalize 0-255 so it looks like a black-and-white photo
            # This makes the "High" blocks white and "Low" blocks black (or vice versa)
            disp = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Contrast Boosting (Optional: makes corners sharper)
            # This is like 'HDR' for your depth map
            disp = cv2.convertScaleAbs(disp, alpha=1.5, beta=0)

            # 3. Visualization
            cv2.imshow("Calibration Stream (Depth Geometry)", disp)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the image for calibration
                filename = f"calib_depth_{frame_count:03d}.png"
                cv2.imwrite(filename, disp)
                print(f"Saved {filename}")
                frame_count += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()