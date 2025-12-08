import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from orbbec_camera import OrbbecCamera # Your working depth-only class

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLOR_CAMERA_INDEX = 0 
CONVERSION_FACTOR = 250000.0 

# --- Camera Intrinsics (FOR 640x480) ---
# We must use the intrinsics for a 640x480 image
# These are typical values, they should be close enough
FX = 570.34
FY = 570.34
CX = 320.0
CY = 240.0
# ------------------------------------

# --- Function to get 3D point ---
def get_world_coords(cx, cy, z_mm, fx, fy, cx_p, cy_p):
    """Converts 2D pixel + Z-depth(mm) to 3D world (X,Y,Z in mm)"""
    if z_mm <= 0 or z_mm > 10000: # Filter invalid depth
        return None
    
    world_x = (cx - cx_p) * z_mm / fx
    world_y = (cy - cy_p) * z_mm / fy
    world_z = z_mm
    return np.array([world_x, world_y, world_z])

# --- convert_disparity_to_depth (Unchanged) ---
def convert_disparity_to_depth(raw_image):
    depth_float = raw_image.astype(np.float32)
    valid_mask = depth_float > 0
    converted_depth = np.zeros_like(depth_float)
    np.divide(CONVERSION_FACTOR, depth_float, out=converted_depth, where=valid_mask)
    return converted_depth

# --- PointSmoother (Unchanged, for Z-value) ---
class PointSmoother:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.data = {}
    def update(self, landmark_idx, new_value):
        if landmark_idx not in self.data:
            self.data[landmark_idx] = deque(maxlen=self.history_size)
        self.data[landmark_idx].append(new_value)
        return int(sum(self.data[landmark_idx]) / len(self.data[landmark_idx]))

def main():
    depth_cam = None
    color_cam = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    smoother = PointSmoother(history_size=10)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            depth_cam = OrbbecCamera()
            color_cam = cv2.VideoCapture(COLOR_CAMERA_INDEX)
            if not color_cam.isOpened():
                raise RuntimeError(f"Could not open color camera at index {COLOR_CAMERA_INDEX}")

            # --- CRITICAL FIX: Match Resolutions ---
            color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # ---------------------------------------

            # Check if it worked
            w_color = color_cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            h_color = color_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Streams initialized. Color: {w_color}x{h_color}, Depth: 640x480")

            while True:
                # 1. Read Frames (both are 640x480)
                raw_depth_image = depth_cam.read_depth_frame()
                ret, color_image_bgr = color_cam.read()

                if not ret:
                    print("Failed to grab color frame.")
                    break
                
                # 2. Flip both for mirror view
                color_image_bgr = cv2.flip(color_image_bgr, 1)
                raw_depth_image = cv2.flip(raw_depth_image, 1)

                # 3. Convert disparity to real millimeters
                true_depth_map = convert_disparity_to_depth(raw_depth_image)

                # 4. MediaPipe Processing
                color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(color_image_rgb)
                
                if results.pose_landmarks:
                    h, w, _ = color_image_bgr.shape # h=480, w=640
                    
                    # 5. Get 3D coordinate for Left Shoulder
                    lm = results.pose_landmarks.landmark
                    ls_landmark = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    
                    cx = int(ls_landmark.x * w)
                    cy = int(ls_landmark.y * h)

                    if 0 <= cx < w and 0 <= cy < h:
                        # Get stable Z value
                        z_mm = true_depth_map[cy, cx]
                        smooth_z_mm = smoother.update(mp_pose.PoseLandmark.LEFT_SHOULDER, z_mm)
                        
                        # Get 3D world point
                        world_point = get_world_coords(cx, cy, smooth_z_mm, FX, FY, CX, CY)
                        
                        if world_point is not None:
                            # 6. PRINT to terminal and DRAW on screen
                            coord_text = f"X: {world_point[0]:.0f} Y: {world_point[1]:.0f} Z: {world_point[2]:.0f} mm"
                            print(f"Left Shoulder 3D: {coord_text}", end="\r") # Print to terminal
                            
                            cv2.putText(color_image_bgr, coord_text, (cx - 100, cy - 20), font, 0.5, (0, 255, 0), 2)
                    
                    mp_drawing.draw_landmarks(color_image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Visualization
                depth_viz = cv2.normalize(true_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

                cv2.imshow("3D Landmarks (640x480)", color_image_bgr)
                cv2.imshow("Depth Map", depth_viz)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if depth_cam: depth_cam.close()
            if color_cam: color_cam.release()
            cv2.destroyAllWindows()
            print("\nDone.")

if __name__ == '__main__':
    main()