# Neural Network Object Detection Library - Interface Specification

## Overview

Simple API for thermal image object detection. User provides grayscale image + IMU data, library handles all preprocessing, inference, and post-processing internally.

---

## API

### Initialization

```python
from detection_library import ObjectDetector

detector = ObjectDetector(
    model_path="/path/to/model_int8.onnx",
    confidence_threshold=0.5,
    max_detections=10
)
```

**Parameters:**
- `model_path` (str): Path to ONNX model file
- `confidence_threshold` (float, optional): Minimum confidence (0.0-1.0). Default: 0.5
- `max_detections` (int, optional): Max detections to return. Default: 10

---

### Detection

```python
detections = detector.detect(
    image,              # numpy.ndarray: grayscale (H, W), uint8, any resolution
    imu_data,           # dict: IMU information
    timestamp           # float: timestamp in seconds
)
```

**Parameters:**
- `image` (numpy.ndarray): Grayscale (H, W), uint8, any resolution
- `imu_data` (dict): IMU data with keys:
  ```python
  {
      "roll": 0.0,      # float: roll angle in degrees
      "pitch": 0.0,     # float: pitch angle in degrees
      "yaw": 0.0        # float: yaw angle in degrees
  }
  ```
- `timestamp` (float): Timestamp in seconds

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

detector = ObjectDetector("model_int8.onnx")

image = cv2.imread("thermal.jpg", cv2.IMREAD_GRAYSCALE)
imu_data = {"roll": 1.2, "pitch": -0.5, "yaw": 45.0}
timestamp = 1234567890.123

detections = detector.detect(image, imu_data, timestamp)

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

detector = ObjectDetector("model_int8.onnx", confidence_threshold=0.6)

cap = cv2.VideoCapture("thermal_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get IMU data from your IMU source
    imu_data = get_current_imu()  # Your function
    timestamp = time.time()

    detections = detector.detect(frame, imu_data, timestamp)

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

### Performance Target
- < 50ms per detection on CPU
- > 20 FPS for video (640×480)

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
