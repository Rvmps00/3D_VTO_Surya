import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from orbbec_camera import OrbbecCamera
import textwrap

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLOR_CAMERA_INDEX = 0 
CONVERSION_FACTOR = 250000.0 
CLOTHING_PATH = "/home/tugasakhir/3D_Camera_VTO/TA_smartMirror/Baju 2D/over-man/"

# --- 2D Landmark Smoother (EMA) ---
# This is still needed to fix the jitter
class LandmarkSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smooth_landmarks = {}
    def __call__(self, raw_landmarks_dict):
        for key, (x, y) in raw_landmarks_dict.items():
            if key not in self.smooth_landmarks:
                self.smooth_landmarks[key] = np.array([x, y], dtype=np.float32)
            else:
                old_pos = self.smooth_landmarks[key]
                new_pos = np.array([x, y], dtype=np.float32)
                smoothed_pos = (old_pos * (1.0 - self.alpha)) + (new_pos * self.alpha)
                self.smooth_landmarks[key] = smoothed_pos
        return self.smooth_landmarks

# --- Helper Function: Overlay with Transparency (Unchanged) ---
def overlay_image(background, foreground):
    if foreground.shape[2] == 4:
        fg_rgb = foreground[:, :, :3]
        alpha_mask = foreground[:, :, 3] / 255.0
        for c in range(3):
            background[:, :, c] = alpha_mask * fg_rgb[:, :, c] + (1 - alpha_mask) * background[:, :, c]
    else:
        background = cv2.addWeighted(background, 1, foreground, 0.5, 0)
    return background

# --- Articulated Sleeve Warping (with anti-tearing fix) ---
def sleeves(frame, sleeve_img, shoulder, elbow, wrist, offset=np.array([0, 0])):
    h_sleeve, w_sleeve = sleeve_img.shape[:2]
    half_h = h_sleeve // 2

    pts_src_upper = np.float32([[0, 0], [w_sleeve, 0], [w_sleeve / 2, half_h]])
    pts_src_lower = np.float32([[0, 0], [w_sleeve, 0], [w_sleeve / 2, half_h]])

    shoulder = (shoulder + offset).astype(np.float32)
    elbow = (elbow + offset).astype(np.float32)
    wrist = (wrist + offset).astype(np.float32)

    # --- FIX FOR TEARING ---
    dir_upper = elbow - shoulder
    norm_upper = np.linalg.norm(dir_upper)
    if norm_upper < 1.0: return frame # Prevent division by zero
        
    dir_lower = wrist - elbow
    norm_lower = np.linalg.norm(dir_lower)
    if norm_lower < 1.0: return frame # Prevent division by zero
    # -----------------------
        
    dir_upper /= norm_upper
    perp_upper = np.array([-dir_upper[1], dir_upper[0]]) * 40

    dir_lower /= norm_lower
    perp_lower = np.array([-dir_lower[1], dir_lower[0]]) * 40

    pts_dst_upper = np.float32([shoulder - perp_upper, shoulder + perp_upper, elbow])
    pts_dst_lower = np.float32([elbow - perp_lower, elbow + perp_lower, wrist])

    upper_half = sleeve_img[0:half_h]
    lower_half = sleeve_img[half_h:]

    M_upper = cv2.getAffineTransform(pts_src_upper, pts_dst_upper)
    warped_upper = cv2.warpAffine(upper_half, M_upper, (frame.shape[1], frame.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    M_lower = cv2.getAffineTransform(pts_src_lower, pts_dst_lower)
    warped_lower = cv2.warpAffine(lower_half, M_lower, (frame.shape[1], frame.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    frame = overlay_image(frame, warped_upper)
    frame = overlay_image(frame, warped_lower)
    return frame

# --- convert_disparity_to_depth (Unchanged) ---
def convert_disparity_to_depth(raw_image):
    depth_float = raw_image.astype(np.float32)
    valid_mask = depth_float > 0
    converted_depth = np.zeros_like(depth_float)
    np.divide(CONVERSION_FACTOR, depth_float, out=converted_depth, where=valid_mask)
    return converted_depth

# --- Text functions (Unchanged, but not called) ---
def roundedRect(img, top_left, bottom_right, color, radius):
    x1, y1 = top_left; x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness=cv2.FILLED)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness=cv2.FILLED)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness=cv2.FILLED)

def textWrap(frame, text, x, y, max_chars_per_line, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1, color_text=(255, 255, 255), color_bg=(0, 0, 0), padding=20, alpha=0.5, max_lines=5, radius=20):
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    if len(wrapped_lines) > max_lines: wrapped_lines = wrapped_lines[:max_lines]
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
    text_width = max(size[0] for size in line_sizes); line_height = line_sizes[0][1] 
    rect_x1 = x - padding; rect_y1 = y - line_height - padding
    rect_x2 = x + text_width + padding; rect_y2 = y + len(wrapped_lines) * line_height + padding
    overlay = frame.copy()
    roundedRect(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color_bg, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(wrapped_lines):
        line_y = y + i * (line_height + 5)
        cv2.putText(frame, line, (x, line_y), font, font_scale, color_text, thickness)
# ------------------------------------

def main():
    depth_cam = None
    color_cam = None
    landmark_smoother = LandmarkSmoother(alpha=0.3) 

    # --- Load Clothing (Unchanged) ---
    try:
        shirt = cv2.imread(CLOTHING_PATH + "dark-denim-mid part.png", cv2.IMREAD_UNCHANGED)
        right_sleeve = cv2.imread(CLOTHING_PATH + "dark-denim-arm.png", cv2.IMREAD_UNCHANGED)
        left_sleeve = cv2.imread(CLOTHING_PATH + "dark-denim-arm-1.png", cv2.IMREAD_UNCHANGED)
        if any(img is None for img in [shirt, right_sleeve, left_sleeve]):
            raise FileNotFoundError("One or more clothing files not found.")
        print("All clothing assets loaded successfully.")
    except Exception as e:
        print(f"--- ERROR: Could not load clothing images ---"); print(f"Details: {e}")
        return
        
    shirt_h, shirt_w, _ = shirt.shape
    pts_src_shirt = np.float32([[shirt_w * 0.2, 0], [shirt_w * 0.8, 0], [shirt_w * 0.7, shirt_h], [shirt_w * 0.3, shirt_h]])
    # -----------------------------------------------

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            depth_cam = OrbbecCamera()
            color_cam = cv2.VideoCapture(COLOR_CAMERA_INDEX)
            
            # --- NEW: FORCE 640x480 RESOLUTION ---
            # This forces the color camera to match the depth camera's resolution.
            color_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            color_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # ---------------------------------------
            
            if not color_cam.isOpened(): raise RuntimeError("Could not open color camera")
            print("Streams initialized. Forcing 640x480. Starting VTO.")

            while True:
                raw_depth_image = depth_cam.read_depth_frame()
                ret, color_image_bgr = color_cam.read()

                if not ret: break
                
                # Check if the resolution change worked
                # if first_frame:
                #    print(f"Color frame size: {color_image_bgr.shape}")
                #    print(f"Depth frame size: {raw_depth_image.shape}")
                #    first_frame = False

                color_image_bgr = cv2.flip(color_image_bgr, 1)
                raw_depth_image = cv2.flip(raw_depth_image, 1)
                true_depth_map = convert_disparity_to_depth(raw_depth_image)

                color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(color_image_rgb)
                
                output_frame = color_image_bgr.copy()
                h, w, _ = output_frame.shape

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    raw_landmarks = {}
                    required_keys = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
                                     "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]
                    
                    for idx, landmark in enumerate(lm):
                        name = mp_pose.PoseLandmark(idx).name
                        if name in required_keys:
                            if landmark.visibility > 0.3:
                                raw_landmarks[name] = (int(landmark.x * w), int(landmark.y * h))
                        
                    smooth_landmarks = landmark_smoother(raw_landmarks)
                    
                    # --- Warp and Overlay (checking for stable points) ---
                    torso_keys = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
                    if all(key in smooth_landmarks for key in torso_keys):
                        offset_shirt = -40
                        pts_dst_shirt = np.float32([smooth_landmarks["LEFT_SHOULDER"] + [0, offset_shirt], 
                                                    smooth_landmarks["RIGHT_SHOULDER"] + [0, offset_shirt], 
                                                    smooth_landmarks["RIGHT_HIP"], 
                                                    smooth_landmarks["LEFT_HIP"]])
                        M_shirt = cv2.getPerspectiveTransform(pts_src_shirt, pts_dst_shirt)
                        warped_shirt = cv2.warpPerspective(shirt, M_shirt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                        output_frame = overlay_image(output_frame, warped_shirt)

                    r_sleeve_keys = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
                    if all(key in smooth_landmarks for key in r_sleeve_keys):
                        output_frame = sleeves(output_frame, right_sleeve, 
                                               smooth_landmarks["RIGHT_SHOULDER"], 
                                               smooth_landmarks["RIGHT_ELBOW"], 
                                               smooth_landmarks["RIGHT_WRIST"], 
                                               offset=np.array([0, -10]))

                    l_sleeve_keys = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
                    if all(key in smooth_landmarks for key in l_sleeve_keys):
                        output_frame = sleeves(output_frame, left_sleeve, 
                                               smooth_landmarks["LEFT_SHOULDER"], 
                                               smooth_landmarks["LEFT_ELBOW"], 
                                               smooth_landmarks["LEFT_WRIST"], 
                                               offset=np.array([0, -10]))
                    
                # --- Visualization ---
                depth_viz = cv2.normalize(true_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

                cv2.imshow("Virtual Try-On (Clean & Stabilized)", output_frame)
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