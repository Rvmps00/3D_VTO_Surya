import ctypes
import numpy as np
import cv2

# --- PATH CONFIGURATION ---
# UPDATE THIS PATH if needed
ORBBEC_LIB_PATH = "/home/tugasakhir/Downloads/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/libOpenNI2.so"

# --- Constants ---
ONI_STATUS_OK = 0
ONI_SENSOR_IR = 3          # <--- NEW: IR Sensor ID
ONI_SENSOR_DEPTH = 1
ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
ONI_PIXEL_FORMAT_GRAY16 = 202 # <--- NEW: IR usually comes as 16-bit Gray
ONI_STREAM_PROPERTY_VIDEO_MODE = 1
ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION = 5 # <--- NEW: Registration ID

# --- Wrapper Setup ---
try:
    openni_lib = ctypes.CDLL(ORBBEC_LIB_PATH)
except OSError as e:
    raise RuntimeError(f"Could not load OpenNI library. Check path: {e}")

class OniVideoMode(ctypes.Structure):
    _fields_ = [("pixelFormat", ctypes.c_int), ("resolutionX", ctypes.c_int), 
                ("resolutionY", ctypes.c_int), ("fps", ctypes.c_int)]

class OniFrame(ctypes.Structure):
    _fields_ = [("dataSize", ctypes.c_int), ("data", ctypes.c_void_p), 
                ("sensorType", ctypes.c_int), ("timestamp", ctypes.c_uint64), 
                ("frameIndex", ctypes.c_int), ("width", ctypes.c_int), 
                ("height", ctypes.c_int), ("videoMode", OniVideoMode), 
                ("croppingEnabled", ctypes.c_bool), ("cropOriginX", ctypes.c_int), 
                ("cropOriginY", ctypes.c_int), ("stride", ctypes.c_int)]

# --- Define Argument Types (Vital for Stability) ---
openni_lib.oniInitialize.restype = ctypes.c_int
openni_lib.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
openni_lib.oniDeviceCreateStream.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
openni_lib.oniStreamStart.argtypes = [ctypes.c_void_p]
openni_lib.oniStreamReadFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(OniFrame))]
openni_lib.oniFrameRelease.argtypes = [ctypes.POINTER(OniFrame)]
openni_lib.oniStreamSetProperty.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
openni_lib.oniDeviceSetProperty.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int] # New

class OrbbecCamera:
    def __init__(self, use_ir=False, enable_registration=True):
        self.device = ctypes.c_void_p()
        self.stream = ctypes.c_void_p() # Can be Depth or IR
        self.frame = ctypes.POINTER(OniFrame)()
        self.use_ir = use_ir

        if openni_lib.oniInitialize(0) != ONI_STATUS_OK:
            raise RuntimeError("OpenNI init failed")

        if openni_lib.oniDeviceOpen(None, ctypes.byref(self.device)) != ONI_STATUS_OK:
            raise RuntimeError("Device open failed")

        # --- ENABLE HARDWARE REGISTRATION (Fixes the Shadow!) ---
        if enable_registration and not use_ir:
            # 1 = ON, 0 = OFF (aligns depth to color sensor)
            mode = ctypes.c_int(1) 
            openni_lib.oniDeviceSetProperty(self.device, ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION, ctypes.byref(mode), 4)
            print("[System] Hardware Depth-to-Color Registration: ENABLED")

        # --- Create Stream (IR or Depth) ---
        sensor_type = ONI_SENSOR_IR if use_ir else ONI_SENSOR_DEPTH
        if openni_lib.oniDeviceCreateStream(self.device, sensor_type, ctypes.byref(self.stream)) != ONI_STATUS_OK:
            raise RuntimeError("Stream creation failed")

        # --- Set Video Mode ---
        # IR needs GRAY16, Depth needs DEPTH_1_MM
        pixel_fmt = ONI_PIXEL_FORMAT_DEPTH_1_MM
        
        mode = OniVideoMode(pixelFormat=pixel_fmt, resolutionX=640, resolutionY=480, fps=30)
        openni_lib.oniStreamSetProperty(self.stream, ONI_STREAM_PROPERTY_VIDEO_MODE, ctypes.byref(mode), ctypes.sizeof(mode))

        openni_lib.oniStreamStart(self.stream)
        print(f"[System] Stream Started. Mode: {'IR (Infrared)' if use_ir else 'Depth (MM)'}")

    def get_frame(self):
        """Returns the frame as a numpy array."""
        openni_lib.oniStreamReadFrame(self.stream, ctypes.byref(self.frame))
        
        # Determine dimensions
        h, w = self.frame.contents.height, self.frame.contents.width
        
        # Copy data buffer to numpy array
        # Both IR and Depth come in as 16-bit integers
        ptr = ctypes.cast(self.frame.contents.data, ctypes.POINTER(ctypes.c_uint16))
        frame_data = np.ctypeslib.as_array(ptr, shape=(h, w)).copy()
        
        openni_lib.oniFrameRelease(self.frame)
        return frame_data

    def close(self):
        # Only try to destroy stream if it was actually created
        if hasattr(self, 'stream') and self.stream: 
            openni_lib.oniStreamDestroy(self.stream)
            self.stream = None # explicit reset
        
        # Only try to close device if it was actually opened
        if hasattr(self, 'device') and self.device: 
            openni_lib.oniDeviceClose(self.device)
            self.device = None
            
        openni_lib.oniShutdown()