import cv2

def main():
    print("--- Searching for IR Camera via Standard Webcam Driver ---")
    print("Press [n] to try the next camera index.")
    print("Press [q] to quit completely.")

    # Try indices 0 to 10
    for index in range(10):
        print(f"\n[Attempting to open Camera Index {index}...] ", end="")
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print("FAILED (No device found)")
            continue
        
        print("SUCCESS!")
        print(f"Showing Camera {index}. Look for Black & White IR video.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream disconnected.")
                break
                
            # Add text label
            cv2.putText(frame, f"Camera Index: {index}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f"Testing Camera {index}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'): # Next
                break
            if key == ord('q'): # Quit
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()