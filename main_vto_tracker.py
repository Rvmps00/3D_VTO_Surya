import cv2
import numpy as np
import mediapipe as mp
from orbbec_camera import OrbbecCamera

# --- 1. CONFIGURATION ---
FX = 600
FY = 600
# Start with 0.6, but we will tune this live
DEPTH_SCALE = 5.15 


# SMOOTHING FACTOR (0.1 = Very Smooth/Slow, 0.9 = Fast/Jittery)
# 0.3 is a good balance for VTO
ALPHA = 0.3 

class ValueSmoother:
    """Simple Exponential Moving Average (EMA) filter"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = (self.alpha * new_value) + ((1 - self.alpha) * self.value)
        return self.value

def get_safe_depth_roi(depth_data, x, y, size=5):
    """
    Reads a square region (ROI) around the point and returns the MEDIAN depth.
    This rejects noise (salt-and-pepper) much better than a single pixel read.
    """
    h, w = depth_data.shape
    x, y = int(x), int(y)
    
    # Define bounds
    x1 = max(0, x - size // 2)
    y1 = max(0, y - size // 2)
    x2 = min(w, x + size // 2)
    y2 = min(h, y + size // 2)
    
    # Extract region
    roi = depth_data[y1:y2, x1:x2]
    
    # Filter out 0s (invalid depth)
    valid_pixels = roi[roi > 0]
    
    if len(valid_pixels) == 0:
        return 0
    
    # Return Median (better than average for noise)
    return np.median(valid_pixels)

def main():
    try:
        cam = OrbbecCamera(use_ir=False, enable_registration=True)
    except Exception as e:
        print(f"Camera Error: {e}")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize Smoothers
    depth_smoother = ValueSmoother(alpha=ALPHA)
    width_smoother = ValueSmoother(alpha=ALPHA)

    print("--- STABILIZED TRACKER V2 ---")
    print(f"Current Scale: {DEPTH_SCALE}")
    print("Press [q] to quit.")

    while True:
        depth_data = cam.get_frame()
        
        # Visualization
        depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        visual_image = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

        results = pose.process(visual_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = visual_image.shape

            # Shoulders
            left_sh = landmarks[11]
            right_sh = landmarks[12]
            
            l_px = (int(left_sh.x * w), int(left_sh.y * h))
            r_px = (int(right_sh.x * w), int(right_sh.y * h))
            cx, cy = (l_px[0] + r_px[0]) // 2, (l_px[1] + r_px[1]) // 2

            # 1. READ DEPTH (Using ROI Median, not single pixel)
            raw_depth = get_safe_depth_roi(depth_data, cx, cy, size=10)

            if raw_depth > 0:
                # 2. APPLY SCALE & SMOOTHING
                # Adjust this math: Raw * Scale
                current_depth_mm = raw_depth * DEPTH_SCALE
                smooth_depth = depth_smoother.update(current_depth_mm)

                # 3. CALCULATE WIDTH
                pixel_width = np.linalg.norm(np.array(l_px) - np.array(r_px))
                
                # Formula: Real = (Pixel * Depth) / Focal
                current_width_mm = (pixel_width * smooth_depth) / FX
                smooth_width = width_smoother.update(current_width_mm)

                # VISUALIZE
                cv2.line(visual_image, l_px, r_px, (0, 255, 255), 3)
                cv2.circle(visual_image, (cx, cy), 5, (0,0,255), -1)

                # Display Data
                cv2.putText(visual_image, f"Depth (Smoothed): {smooth_depth:.0f} mm", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show Raw Depth too for debugging
                cv2.putText(visual_image, f"Raw Sensor: {raw_depth:.0f}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.putText(visual_image, f"Shoulder Width: {smooth_width:.0f} mm", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(visual_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("VTO Tracker V2", visual_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()