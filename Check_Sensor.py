import ctypes

# --- PATH (Same as before) ---
ORBBEC_LIB_PATH = "/home/tugasakhir/Downloads/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/libOpenNI2.so"

# --- Constants ---
ONI_STATUS_OK = 0

try:
    openni_lib = ctypes.CDLL(ORBBEC_LIB_PATH)
except OSError as e:
    print(f"Library Load Error: {e}")
    exit(1)

# --- Types ---
class OniSensorInfo(ctypes.Structure):
    _fields_ = [("sensorType", ctypes.c_int),
                ("numSupportedVideoModes", ctypes.c_int),
                ("pSupportedVideoModes", ctypes.c_void_p)]

class OniDeviceInfo(ctypes.Structure):
    _fields_ = [("uri", ctypes.c_char * 256),
                ("vendor", ctypes.c_char * 256),
                ("name", ctypes.c_char * 256),
                ("usbVendorId", ctypes.c_uint16),
                ("usbProductId", ctypes.c_uint16)]

# --- Bindings ---
openni_lib.oniInitialize.restype = ctypes.c_int
openni_lib.oniGetDeviceList.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
openni_lib.oniDeviceOpen.restype = ctypes.c_int
openni_lib.oniDeviceGetSensorInfo.restype = ctypes.POINTER(OniSensorInfo)

def main():
    print("--- Checking Orbbec Sensors ---")
    openni_lib.oniInitialize(0)
    
    device_list = ctypes.c_void_p()
    num_devices = ctypes.c_int()
    openni_lib.oniGetDeviceList(ctypes.byref(device_list), ctypes.byref(num_devices))
    
    if num_devices.value == 0:
        print("No devices found!")
        return

    print(f"Found {num_devices.value} device(s). Opening the first one...")
    
    device = ctypes.c_void_p()
    ret = openni_lib.oniDeviceOpen(None, ctypes.byref(device))
    if ret != ONI_STATUS_OK:
        print("Failed to open device.")
        return

    # Check for Sensor Types 1 to 10
    sensor_names = {1: "DEPTH", 2: "COLOR", 3: "IR", 4: "FISHEYE"}
    
    print("\nSupported Sensors:")
    for i in range(1, 10):
        info_ptr = openni_lib.oniDeviceGetSensorInfo(device, i)
        if info_ptr:
            name = sensor_names.get(info_ptr.contents.sensorType, "UNKNOWN")
            print(f" - Sensor ID {info_ptr.contents.sensorType}: {name} (AVAILABLE)")
        else:
            pass # Sensor ID not found

    openni_lib.oniDeviceClose(device)
    openni_lib.oniShutdown()

if __name__ == "__main__":
    main()