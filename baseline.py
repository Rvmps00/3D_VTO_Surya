import cv2
import numpy as np
from orbbec_camera import OrbbecCamera 

COLOR_CAMERA_INDEX = 0 

def main():
    depth_cam = None
    color_cam = None
    try:
        depth_cam = OrbbecCamera()
        color_cam = cv2.VideoCapture(COLOR_CAMERA_INDEX)
        if not color_cam.isOpened():
            raise RuntimeError(f"Could not open color camera at index {COLOR_CAMERA_INDEX}")

        print("Both streams initialized. Press 'q' to quit.")

        while True:
            depth_image = depth_cam.read_depth_frame()
            ret, color_image_bgr = color_cam.read()

            if not ret:
                print("Failed to grab color frame.")
                break
                
            # --- FIX: FLIP THE MIRRORED IMAGE ---
            color_image_bgr = cv2.flip(color_image_bgr, 1)

            # 1. Prepare Depth Visualization (Color Map)
            depth_viz = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

            # 2. Resize Depth to match Color Dimensions
            # Operations like addWeighted fail if shapes (width/height) don't match exactly.
            h, w = color_image_bgr.shape[:2]
            depth_viz_resized = cv2.resize(depth_viz, (w, h))

            # 3. Blend the images
            # Arguments: (Img1, Alpha, Img2, Beta, Gamma)
            # Change 0.6/0.4 to adjust transparency.
            overlay = cv2.addWeighted(color_image_bgr, 0.6, depth_viz_resized, 0.4, 0)

            # 4. Display
            cv2.imshow("Overlay (Color + Depth)", overlay)
            # Optional: Show raw feeds separately if you want
            # cv2.imshow("Depth Only", depth_viz)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if depth_cam:
            depth_cam.close()
        if color_cam:
            color_cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()