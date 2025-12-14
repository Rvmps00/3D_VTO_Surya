import cv2
import numpy as np
import mediapipe as mp
import os
from orbbec_camera import OrbbecCamera

# --- 1. CONFIGURATION ---
FX = 1007.27
FY = 996.78
DEPTH_SCALE = 5.15

# ALIGNMENT 
SHIFT_X = 29
SHIFT_Y = -21

# --- FIT SETTINGS (TWEAK THESE!) ---
SHIRT_SCALE = 1.3      # 1.0 = Exact shoulder width, 1.5 = Loose Fit
VERTICAL_OFFSET = 45   # Positive = Move Shirt UP, Negative = Move DOWN

# ASSET PATHS
ASSET_DIR = "assets"
TORSO_IMG = "torso.png" 

class VirtualTryOnV2:
    def __init__(self):
        self.torso = self.load_asset(TORSO_IMG)
        
        try:
            self.depth_cam = OrbbecCamera(use_ir=False, enable_registration=True)
        except:
            print("Error: Connect Depth Camera")
            return
        
        self.rgb_cam = cv2.VideoCapture(0) 
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def load_asset(self, name):
        path = os.path.join(ASSET_DIR, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        if img is None:
            print(f"[WARNING] Could not load {name}. Creating placeholder.")
            img = np.zeros((200, 200, 4), dtype=np.uint8)
            img[:] = (0, 255, 0, 255) 
        return img

    def overlay_transparent(self, background, foreground, x, y):
        """
        Pastes 'foreground' onto 'background' at (x,y) handling alpha blending.
        """
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]

        # Check if the image is completely outside the screen
        if x >= bg_w or y >= bg_h or x + fg_w <= 0 or y + fg_h <= 0:
            return background

        # Calculate intersection clipping
        # Example: if x is -10 (offscreen left), we crop the first 10px of the foreground
        fg_x_start = max(0, -x)
        fg_y_start = max(0, -y)
        fg_x_end = min(fg_w, bg_w - x)
        fg_y_end = min(fg_h, bg_h - y)

        bg_x_start = max(0, x)
        bg_y_start = max(0, y)
        bg_x_end = min(bg_w, x + fg_w)
        bg_y_end = min(bg_h, y + fg_h)

        # Extract the overlapping regions
        fg_crop = foreground[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
        bg_crop = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end]

        # Safety check for empty slice
        if fg_crop.shape[0] == 0 or fg_crop.shape[1] == 0:
            return background

        # Separate Alpha and Color
        alpha_s = fg_crop[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # Blend for B, G, R
        for c in range(0, 3):
            bg_crop[:, :, c] = (alpha_s * fg_crop[:, :, c] +
                                alpha_l * bg_crop[:, :, c])

        # Put back
        background[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = bg_crop
        return background

    def run(self):
        print("--- VTO V2 (SCALING MODE) ---")
        print(f"Shift: {SHIFT_X}, {SHIFT_Y}")
        print("Press [q] to Quit.")

        while True:
            depth_data = self.depth_cam.get_frame()
            ret, rgb_frame = self.rgb_cam.read()
            if not ret: break

            h, w, _ = rgb_frame.shape
            
            # Run Tracking
            rgb_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_rgb)

            canvas = rgb_frame.copy()

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Get Shoulders
                l_sh = np.array([lm[11].x * w, lm[11].y * h])
                r_sh = np.array([lm[12].x * w, lm[12].y * h])
                
                # 1. CALCULATE WIDTH & CENTER
                # Distance between shoulders in pixels
                shoulder_width_px = np.linalg.norm(l_sh - r_sh)
                
                # Midpoint between shoulders
                center_x = int((l_sh[0] + r_sh[0]) / 2)
                center_y = int((l_sh[1] + r_sh[1]) / 2)

                # 2. RESIZE SHIRT
                # How wide is the PNG originally?
                orig_h, orig_w = self.torso.shape[:2]
                
                # We want the shirt to be (Shoulder Width * Scale Factor) wide
                target_width = int(shoulder_width_px * SHIRT_SCALE)
                
                # Calculate scale ratio to keep aspect ratio correct
                ratio = target_width / float(orig_w)
                target_height = int(orig_h * ratio)
                
                # Resize
                if target_width > 0 and target_height > 0:
                    try:
                        resized_shirt = cv2.resize(self.torso, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        
                        # 3. CALCULATE POSITION (Top-Left corner for pasting)
                        # We want 'center_x' to be the middle of the shirt
                        paste_x = int(center_x - (target_width / 2))
                        
                        # We want 'center_y' to be near the neck (Top of shirt)
                        # Apply Vertical Offset
                        paste_y = int(center_y - (VERTICAL_OFFSET * ratio)) 

                        # 4. OVERLAY
                        canvas = self.overlay_transparent(canvas, resized_shirt, paste_x, paste_y)
                        
                        # Debug: Draw Green Dot at Chest Center
                        cv2.circle(canvas, (center_x, center_y), 5, (0, 255, 0), -1)
                        
                    except Exception as e:
                        pass # Ignore resize errors on bad frames

            cv2.imshow("VTO Final V2", canvas)
            if cv2.waitKey(1) == ord('q'):
                break

        self.depth_cam.close()
        self.rgb_cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VirtualTryOnV2()
    app.run()