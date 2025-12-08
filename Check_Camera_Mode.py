import ctypes

# --- PATH (Keep your existing path) ---
ORBBEC_LIB_PATH = "/home/tugasakhir/Downloads/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/libOpenNI2.so"

ONI_STATUS_OK = 0
ONI_SENSOR_IR = 3  # The ID we confirmed exists

try:
    openni_lib = ctypes.CDLL(ORBBEC_LIB_PATH)
except OSError as e:
    print(f"Library Error: {e}")
    exit(1)

# --- Structures ---
class OniVideoMode(ctypes.Structure):
    _fields_ = [("pixelFormat", ctypes.c_int), 
                ("resolutionX", ctypes.c_int), 
                ("resolutionY", ctypes.c_int), 
                ("fps", ctypes.c_int)]

class OniSensorInfo(ctypes.Structure):
    _fields_ = [("sensorType", ctypes.c_int),
                ("numSupportedVideoModes", ctypes.c_int),
                ("pSupportedVideoModes", ctypes.POINTER(OniVideoMode))]

# --- Bindings ---
openni_lib.oniInitialize.restype = ctypes.c_int
openni_lib.oniDeviceOpen.restype = ctypes.c_int
openni_lib.oniDeviceGetSensorInfo.restype = ctypes.POINTER(OniSensorInfo)

def main():
    openni_lib.oniInitialize(0)
    device = ctypes.c_void_p()
    if openni_lib.oniDeviceOpen(None, ctypes.byref(device)) != ONI_STATUS_OK:
        print("Failed to open device.")
        return

    print(f"--- Checking Video Modes for Sensor ID {ONI_SENSOR_IR} (IR) ---")
    
    info_ptr = openni_lib.oniDeviceGetSensorInfo(device, ONI_SENSOR_IR)
    
    if not info_ptr:
        print("Could not get sensor info. (Is the ID correct?)")
    else:
        count = info_ptr.contents.numSupportedVideoModes
        modes = info_ptr.contents.pSupportedVideoModes
        
        pixel_format_names = {
            100: "DEPTH_1MM",
            101: "DEPTH_100UM",
            102: "SHIFT_9_2",
            103: "SHIFT_9_3",
            200: "RGB888",
            201: "YUV422",
            202: "GRAY16",
            203: "GRAY8", 
            204: "JPEG"
        }

        print(f"Found {count} supported modes:")
        print(f"{'INDEX':<6} | {'RES_X':<6} | {'RES_Y':<6} | {'FPS':<4} | {'FORMAT'}")
        print("-" * 50)
        
        for i in range(count):
            m = modes[i]
            fmt_name = pixel_format_names.get(m.pixelFormat, f"UNKNOWN({m.pixelFormat})")
            print(f"{i:<6} | {m.resolutionX:<6} | {m.resolutionY:<6} | {m.fps:<4} | {fmt_name}")

    openni_lib.oniDeviceClose(device)
    openni_lib.oniShutdown()

if __name__ == "__main__":
    main()