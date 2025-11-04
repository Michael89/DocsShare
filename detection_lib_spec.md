# Neural Network Object Detection Library - Interface Specification

## Overview

Simple API for thermal image object detection. User provides grayscale image + IMU data, library handles all preprocessing, inference, and post-processing internally.

---

## API

### Initialization

```python
from detection_library import ObjectDetector

# Constructor (exception-free)
detector = ObjectDetector(
    confidence_threshold=0.5,
    max_detections=10
)

# Load model (can throw ModelLoadError)
detector.load_model("/path/to/model_int8.onnx")
```

**Constructor Parameters:**
- `confidence_threshold` (float, optional): Minimum confidence (0.0-1.0). Default: 0.5
- `max_detections` (int, optional): Max detections to return. Default: 10

**load_model Parameters:**
- `model_path` (str): Path to ONNX model file. Throws `ModelLoadError` if file doesn't exist or model is invalid.

---

### Detection

```python
detections = detector.detect(
    image,              # numpy.ndarray: grayscale (H, W), uint8, any resolution
    frame_timestamp,    # float: frame timestamp in seconds
    attitude,           # dict: attitude information
    attitude_timestamp  # float: attitude timestamp in seconds
)
```

**Parameters:**
- `image` (numpy.ndarray): Grayscale (H, W), uint8, any resolution
- `frame_timestamp` (float): Frame timestamp in seconds
- `attitude` (dict): Attitude data with keys:
  ```python
  {
      "roll": 0.0,      # float: roll angle in radians
      "pitch": 0.0,     # float: pitch angle in radians
      "yaw": 0.0        # float: yaw angle in radians
  }
  ```
- `attitude_timestamp` (float): Attitude timestamp in seconds

**Returns:**
```python
[
    [x, y, confidence],    # [float, float, float]: x, y in original image coords, confidence (0.0-1.0)
    [x2, y2, confidence2],
    ...
]
```

**Array format:** Each detection is `[x, y, confidence]`
- `x` (float): X coordinate of center point in original image
- `y` (float): Y coordinate of center point in original image
- `confidence` (float): Detection confidence score (0.0-1.0)

---

## Internal Processing (What Library Must Handle)

### Preprocessing
1. Validate image (single channel grayscale, uint8)
2. Resize to model input dimensions (e.g., 480×384)
3. Normalize to [0.0, 1.0]
4. Reshape to (1, 1, H, W) for ONNX

### Inference
Run ONNX model, extract heatmap output

### Post-processing
1. Find local maxima in heatmap
2. Filter by confidence threshold
3. Transform coordinates back to original image resolution
4. Sort by confidence, limit to max_detections

---

## Usage Examples

### Basic Detection

```python
from detection_library import ObjectDetector
import cv2

detector = ObjectDetector()
detector.load_model("model_int8.onnx")

image = cv2.imread("thermal.jpg", cv2.IMREAD_GRAYSCALE)
frame_timestamp = 1234567890.123
attitude = {"roll": 0.021, "pitch": -0.009, "yaw": 0.785}  # radians
attitude_timestamp = 1234567890.120

detections = detector.detect(image, frame_timestamp, attitude, attitude_timestamp)

for det in detections:
    x, y, confidence = det
    print(f"Object at ({x:.1f}, {y:.1f}), confidence: {confidence:.2f}")
    cv2.circle(image, (int(x), int(y)), 5, 255, -1)

cv2.imshow("Detections", image)
cv2.waitKey(0)
```

### Video Processing

```python
from detection_library import ObjectDetector
import cv2
import time

detector = ObjectDetector(confidence_threshold=0.6)
detector.load_model("model_int8.onnx")

cap = cv2.VideoCapture("thermal_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get frame and attitude with timestamps
    frame_timestamp = time.time()
    attitude = get_current_attitude()  # Your function: returns {"roll": ..., "pitch": ..., "yaw": ...} in radians
    attitude_timestamp = get_attitude_timestamp()  # Your function

    detections = detector.detect(frame, frame_timestamp, attitude, attitude_timestamp)

    for det in detections:
        x, y, confidence = det
        cv2.circle(frame, (int(x), int(y)), 5, 255, -1)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Implementation Requirements

### Language Options
- **Python** (recommended): `numpy`, `opencv-python`, `onnxruntime`
- **C++** (for performance): OpenCV C++, ONNX Runtime C++ API, Python bindings via pybind11

### Dependencies
- Python ≥ 3.9
- numpy ≥ 1.24.3
- opencv-python ≥ 4.10.0
- onnxruntime ≥ 1.18.1

---

## C++ Implementation (with pybind11)

If implementing in C++, the API should provide the same interface:

### C++ Header (detector.h)

```cpp
#include <vector>
#include <string>

namespace detection_library {

struct AttitudeRad {
    double roll;   // radians
    double pitch;  // radians
    double yaw;    // radians
};

class ObjectDetector {
public:
    // Constructor (exception-free)
    ObjectDetector(double confidence_threshold = 0.5,
                   int max_detections = 10);

    ~ObjectDetector();

    // Load model (can throw exception)
    void load_model(const std::string& model_path);

    // Returns vector of detections: [[x, y, confidence], ...]
    std::vector<std::vector<double>> detect(
        const uint8_t* image_data,       // Grayscale image data
        size_t width,                    // Image width
        size_t height,                   // Image height
        size_t stride,                   // Row stride (bytes per row, usually == width for grayscale)
        double frame_timestamp,          // Frame timestamp in seconds
        const AttitudeRad& attitude,     // Attitude angles in radians
        double attitude_timestamp        // Attitude timestamp in seconds
    );

private:
    void* ort_session_;  // ONNX Runtime session
    double confidence_threshold_;
    int max_detections_;
    int model_input_width_;
    int model_input_height_;
};

}  // namespace detection_library
```

### Python Bindings (bindings.cpp)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "detector.h"

namespace py = pybind11;
using namespace detection_library;

// Wrapper to accept numpy array
std::vector<std::vector<double>> detect_wrapper(
    ObjectDetector& self,
    py::array_t<uint8_t> image,
    double frame_timestamp,
    py::dict attitude,
    double attitude_timestamp
) {
    // Extract numpy array
    auto buf = image.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be 2D grayscale array");
    }

    // Get image data pointer and dimensions
    const uint8_t* image_data = static_cast<const uint8_t*>(buf.ptr);
    size_t height = buf.shape[0];
    size_t width = buf.shape[1];
    size_t stride = buf.strides[0];  // Bytes per row

    // Extract attitude data from dict
    AttitudeRad att;
    att.roll = attitude["roll"].cast<double>();
    att.pitch = attitude["pitch"].cast<double>();
    att.yaw = attitude["yaw"].cast<double>();

    // Call C++ detect
    return self.detect(image_data, width, height, stride, frame_timestamp, att, attitude_timestamp);
}

PYBIND11_MODULE(detection_library, m) {
    m.doc() = "Object detection library with ONNX model";

    py::class_<ObjectDetector>(m, "ObjectDetector")
        .def(py::init<double, int>(),
             py::arg("confidence_threshold") = 0.5,
             py::arg("max_detections") = 10,
             "Constructor (exception-free)")
        .def("load_model", &ObjectDetector::load_model,
             py::arg("model_path"),
             "Load ONNX model (can throw exception)")
        .def("detect", &detect_wrapper,
             py::arg("image"),
             py::arg("frame_timestamp"),
             py::arg("attitude"),
             py::arg("attitude_timestamp"),
             "Detect objects in image");
}
```

**Key Points:**
- Constructor is exception-free, only sets parameters
- `load_model()` handles file I/O and can throw exceptions
- Image data passed as standard pointer: `uint8_t* image_data, size_t width, size_t height, size_t stride`
- Two timestamps: `frame_timestamp` and `attitude_timestamp`
- pybind11 wrapper extracts numpy array pointer and dimensions
- Python dict for attitude is converted to C++ `AttitudeRad` struct (angles in radians)
- Returns `std::vector<std::vector<double>>` (auto-converts to Python list of lists)

---

## Error Handling

```python
class DetectionError(Exception):
    """Base exception"""
    pass

class ModelLoadError(DetectionError):
    """Error loading ONNX model"""
    pass

class InvalidImageError(DetectionError):
    """Invalid input image"""
    pass
```

**Behavior:**
- Invalid model path → `ModelLoadError`
- Multi-channel image or wrong type → `InvalidImageError`
- Empty detections → Return empty list

---

## Deliverables

```
detection_library/
├── __init__.py
├── detector.py
├── preprocessing.py
├── postprocessing.py
└── exceptions.py

examples/
├── basic_detection.py
└── video_processing.py

requirements.txt
setup.py or pyproject.toml
README.md
```

**Note:** Implementation can be in pure Python or C++ with Python bindings (pybind11). Both should provide the same API interface.
