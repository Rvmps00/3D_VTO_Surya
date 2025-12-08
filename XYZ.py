import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from orbbec_camera import OrbbecCamera

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLOR_CAMERA_INDEX = 0 

# --- YOUR CAMERA'S CALIBRATED CONVERSION FACTOR ---
# Derived from your data: 1000mm * 250 (raw) = 250,000
CONVERSION_FACTOR = 250000.0 

class PointSmoother:
    """Averages the last 10 valid Z values to reduce jitter."""
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.data = {}
    def update(self, landmark_idx, new_value):
        if landmark_idx not in self.data:
            self.data[landmark_idx] = deque(maxlen=self.history_size)
        self.data[landmark_idx].append(new_value)
        return int(sum(self.data[landmark_idx]) / len(self.data[landmark_idx]))

def convert_disparity_to_depth(raw_image):
    """
    Converts raw disparity (inverted) values to depth in millimeters
    using your calibrated factor.
    """
    depth_float = raw_image.astype(np.float32)
    valid_mask = depth_float > 0
    converted_depth = np.zeros_like(depth_float)
    
    # Formula: Z_mm = Factor / Raw_Disparity
    np.divide(CONVERSION_FACTOR, depth_float, out=converted_depth, where=valid_mask)
    
    return converted_depth

def main():
    depth_cam = None
    color_cam = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    smoother = PointSmoother(history_size=10)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            # Initialize Hybrid Streams
            depth_cam = OrbbecCamera()
            color_cam = cv2.VideoCapture(COLOR_CAMERA_INDEX)
            if not color_cam.isOpened():
                raise RuntimeError(f"Could not open color camera at index {COLOR_CAMERA_INDEX}")

            print("Streams initialized. Using calibrated factor 250,000.")

            while True:
                # 1. Read Frames
                raw_depth_image = depth_cam.read_depth_frame()
                ret, color_image_bgr = color_cam.read()

                if not ret:
                    print("Failed to grab color frame.")
                    break
                
                # 2. Flip both for mirror view
                color_image_bgr = cv2.flip(color_image_bgr, 1)
                # raw_depth_image = cv2.flip(raw_depth_image, 1)

                # 3. Convert disparity to real millimeters
                true_depth_map = convert_disparity_to_depth(raw_depth_image)

                # 4. MediaPipe Processing
                color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(color_image_rgb)
                
                if results.pose_landmarks:
                    h, w, _ = color_image_bgr.shape
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx < 11: continue # Skip head

                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
                        if 0 <= cx < w and 0 <= cy < h:
                            if cy < true_depth_map.shape[0] and cx < true_depth_map.shape[1]:
                                # Read from the CORRECTED depth map
                                real_z_mm = true_depth_map[cy, cx]
                                
                                # Filter invalid values (0 or too far)
                                if real_z_mm <= 0 or real_z_mm > 10000: continue

                                # Smooth the Z value to stop jitter
                                smooth_z_mm = smoother.update(idx, real_z_mm)
                                
                                # Display (X, Y, Z_mm)
                                coord_text = f"({cx}, {cy}, {smooth_z_mm})"
                                cv2.circle(color_image_bgr, (cx, cy), 4, (0, 0, 255), -1)
                                cv2.putText(color_image_bgr, coord_text, (cx + 10, cy), font, 0.4, (0, 255, 255), 1)

                    mp_drawing.draw_landmarks(color_image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Visualization (show the corrected depth map)
                depth_viz = cv2.normalize(true_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

                cv2.imshow("Body XYZ (Calibrated MM)", color_image_bgr)
                cv2.imshow("Depth Map (Corrected)", depth_viz)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if depth_cam: depth_cam.close()
            if color_cam: color_cam.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()