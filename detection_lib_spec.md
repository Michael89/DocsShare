# Neural Network Object Detection Library - Interface Specification

## 1. Overview

**The end-user should only need to:**
1. Initialize the detector with a model path
2. Pass a grayscale image in any resolution
3. Receive structured detection results with exact object positions

All preprocessing (resizing, normalization) and post-processing (heatmap analysis, peak detection, coordinate transformation) are handled internally.

---

## 2. API Design

### 2.1 Initialization

```python
from detection_library import ObjectDetector

# Initialize detector with model path
detector = ObjectDetector(
    model_path="/path/to/model_int8.onnx",
    confidence_threshold=0.5,
    max_detections=10
)
```

**Parameters:**
- `model_path` (str, required): Path to ONNX model file
- `confidence_threshold` (float, optional): Minimum confidence for detection (0.0-1.0). Default: 0.5
- `max_detections` (int, optional): Maximum number of detections to return. Default: 10

**Behavior:**
- Load ONNX model
- Extract model input/output shapes
- Initialize preprocessing/postprocessing pipelines
- Validate model compatibility

---

### 2.2 Detection Method

```python
# Detect objects in a single image
detections = detector.detect(image)
```

**Input Parameters:**
- `image` (numpy.ndarray, required):
  - Format: Grayscale (H, W) - **single channel only**
  - Type: `uint8` (0-255)
  - Resolution: Any size (will be automatically resized internally)

**Return Value:**
```python
# List of Detection objects
[
    Detection(
        position=(x, y),           # Center point in original image coordinates
        confidence=0.87            # Confidence score (0.0-1.0)
    ),
    Detection(
        position=(x2, y2),
        confidence=0.72
    ),
    ...
]
```

---

### 2.3 Detection Object Structure

```python
class Detection:
    """Single object detection result"""

    position: Tuple[float, float]          # (x, y) center in original image coordinates
    confidence: float                      # Detection confidence (0.0-1.0)
```

---

## 3. Internal Processing Pipeline

The library handles these steps internally (transparent to the user):

### 3.1 Preprocessing
1. **Input Validation**
   - Verify image is single channel grayscale (H, W)
   - Verify image type is uint8
   - Raise error if multi-channel image or wrong type provided

2. **Resizing**
   - Resize image to model input dimensions (e.g., 480×384)
   - Store scaling factors for coordinate transformation
   - Use appropriate interpolation (e.g., `cv2.INTER_LINEAR`)

3. **Normalization**
   - Convert pixel values to float32
   - Normalize to [0.0, 1.0] range (divide by 255.0)

4. **Dimension Reshaping**
   - Add batch dimension: (H, W) → (1, 1, H, W)
   - Ensure correct dimension order for ONNX model

### 3.2 Inference
1. Run ONNX model inference
2. Extract output heatmap (shape: [1, 1, H, W])
3. Remove batch dimensions: (1, 1, H, W) → (H, W)

### 3.3 Post-processing
1. **Peak Detection**
   - Find local maxima in heatmap
   - Filter peaks below confidence threshold

2. **Coordinate Transformation**
   - Transform heatmap coordinates to original image coordinates
   - Apply inverse scaling transformation
   - Ensure coordinates are within image bounds

3. **Sorting**
   - Sort detections by confidence (highest first)
   - Limit to `max_detections`

---

## 4. Implementation Requirements

### 4.1 Language Options

**Option 1: Pure Python** (Recommended)
- Easier to develop and maintain
- Good integration with Python ecosystem
- Sufficient performance for most use cases
- Dependencies: `numpy`, `opencv-python`, `onnxruntime`

**Option 2: C++ with Python Bindings** (For performance-critical applications)
- Better performance (especially for post-processing)
- More complex development and deployment
- Use `pybind11` or `ctypes` for Python bindings
- Dependencies: OpenCV C++, ONNX Runtime C++ API

### 4.2 Minimum Dependencies
- Python ≥ 3.9
- `numpy` ≥ 1.24.3
- `opencv-python` ≥ 4.10.0
- `onnxruntime` ≥ 1.18.1

---

## 5. Usage Examples

### 5.1 Basic Usage

```python
from detection_library import ObjectDetector
import cv2

# Initialize detector
detector = ObjectDetector("model_int8.onnx", confidence_threshold=0.5, max_detections=10)

# Load image (must be grayscale, uint8)
image = cv2.imread("thermal_image.jpg", cv2.IMREAD_GRAYSCALE)  # Any resolution

# Detect objects
detections = detector.detect(image)

# Process results
for det in detections:
    print(f"Object at ({det.position[0]:.1f}, {det.position[1]:.1f}), "
          f"confidence: {det.confidence:.2f}")

    # Draw on image
    x, y = det.position
    cv2.circle(image, (int(x), int(y)), 5, 255, -1)
    cv2.putText(image, f"{det.confidence:.2f}", (int(x)+10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
```

### 5.2 Video Processing

```python
from detection_library import ObjectDetector
import cv2

# Initialize detector
detector = ObjectDetector("model_int8.onnx", confidence_threshold=0.6, max_detections=5)

# Open video
cap = cv2.VideoCapture("thermal_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects (frame can be any resolution)
    detections = detector.detect(frame)

    # Visualize
    for det in detections:
        x, y = det.position
        cv2.circle(frame, (int(x), int(y)), 5, 255, -1)

    cv2.imshow("Video Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5.3 Different Configurations

```python
from detection_library import ObjectDetector

# High confidence detections
detector_strict = ObjectDetector("model_int8.onnx", confidence_threshold=0.8, max_detections=5)

# More permissive detections
detector_permissive = ObjectDetector("model_int8.onnx", confidence_threshold=0.3, max_detections=20)

# Detect on grayscale image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
detections = detector_strict.detect(image)
```

---

## 6. Error Handling

The library must handle and report errors gracefully:

```python
class DetectionError(Exception):
    """Base exception for detection errors"""
    pass

class ModelLoadError(DetectionError):
    """Error loading ONNX model"""
    pass

class InvalidImageError(DetectionError):
    """Invalid input image format"""
    pass

class InferenceError(DetectionError):
    """Error during model inference"""
    pass
```

**Expected Behaviors:**
- Invalid model path → `ModelLoadError` with clear message
- Multi-channel image provided → `InvalidImageError`: "Image must be single channel grayscale (H, W)"
- Wrong image type (not uint8) → `InvalidImageError`: "Image must be uint8 type"
- Invalid image format → `InvalidImageError` with format details
- Inference failure → `InferenceError` with model details
- Empty detections → Return empty list (not an error)

---

## 7. Documentation Requirements

The implementation must include:

1. **README.md**
   - Installation instructions
   - Quick start guide
   - API reference

2. **API Documentation**
   - Complete class/method documentation
   - Parameter descriptions
   - Return value specifications
   - Code examples

3. **User Guide**
   - Configuration guide
   - Performance tuning tips
   - Troubleshooting section

---

## 8. Deliverables

### 8.1 Required Files
```
detection_library/
├── __init__.py              # Main module exports
├── detector.py              # ObjectDetector class
├── detection.py             # Detection class
├── preprocessing.py         # Image preprocessing functions
├── postprocessing.py        # Heatmap post-processing functions
└── exceptions.py            # Custom exceptions

examples/
├── basic_detection.py
└── video_processing.py

docs/
├── README.md
└── API.md

requirements.txt             # Python dependencies
setup.py or pyproject.toml  # Package configuration
```

### 8.2 Package Structure (Python)
```python
# Installable package
pip install detection_library

# Import and use
from detection_library import ObjectDetector, Detection
```

### 8.3 Package Structure (C++ with Python bindings)
```
detection_library/
├── src/                     # C++ source files
│   ├── detector.cpp
│   ├── preprocessing.cpp
│   └── postprocessing.cpp
├── include/                 # C++ headers
│   └── detection_library/
│       ├── detector.h
│       └── types.h
├── python/                  # Python bindings
│   ├── __init__.py
│   └── bindings.cpp (pybind11)
└── CMakeLists.txt           # Build configuration
```

---

## 9. Summary

This specification defines a simple, clean interface for neural network-based object detection:

**Input:** Grayscale image (uint8, any resolution)
**Output:** List of detections with position and confidence
**Internal:** All preprocessing, inference, and post-processing handled by the library

The implementation should prioritize simplicity and ease of use while maintaining good performance.

---
