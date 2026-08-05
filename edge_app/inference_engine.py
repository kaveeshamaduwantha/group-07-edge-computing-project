"""
=============================================================================
inference_engine.py — Smart Classroom Edge AI | YOLOv8 Object Detection Engine
=============================================================================
This module handles ALL AI inference operations:
  1. Loads the trained YOLOv8 model (best_classroom_yolo.pt)
  2. Runs object detection on individual video frames
  3. Counts students and identifies janitors
  4. Applies sliding-window smoothing to avoid flickering predictions
  5. Maps student count → occupancy band (LOW / MEDIUM / HIGH)
  6. Exposes a set_thresholds() function for live band boundary updates

Detected Classes:
  Class 0 = student  (green bounding box)
  Class 1 = janitor  (purple bounding box)
=============================================================================
"""

import json, os, time
import cv2
import numpy as np

# =============================================================================
# SECTION 1: MODEL FILE PATHS
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
YOLO_PATH = os.path.join(MODEL_DIR, "best_classroom_yolo.pt")   # YOLOv8 weights
CFG_PATH  = os.path.join(MODEL_DIR, "detector_config.json")      # Config file

# =============================================================================
# SECTION 2: LOAD CONFIGURATION FILE
# =============================================================================
if os.path.exists(CFG_PATH):
    with open(CFG_PATH) as f:
        cfg = json.load(f)
else:
    cfg = {}

CLASS_NAMES = cfg.get("class_names", ["student", "janitor"])
CONF_THRESH = cfg.get("confidence_threshold", 0.45)

# =============================================================================
# SECTION 3: MODEL LOADING
# =============================================================================
_model = None

def _load_model():
    """
    PROCESS: YOLOv8 Model Initialization
    Loads best_classroom_yolo.pt once at startup using ultralytics library.
    Sets _model=None if loading fails so predict_frame() returns safe zeros.
    """
    global _model
    try:
        from ultralytics import YOLO
        _model = YOLO(YOLO_PATH)
        print(f"[inference] YOLOv8 loaded: {YOLO_PATH}")
        print(f"[inference] Classes: {CLASS_NAMES}  Threshold: {CONF_THRESH}")
    except Exception as e:
        print(f"[inference] ERROR loading YOLO: {e}")
        _model = None

_load_model()

# =============================================================================
# SECTION 4: RATE LIMITING & SMOOTHING STATE
# =============================================================================
_last_call   = 0.0      # Timestamp of last inference run
_interval    = 0.15     # Min seconds between inferences (~6-7 FPS max)
_last_result = None     # Cached result returned while rate limited

_vote_buf = []
VOTE_WIN  = 5           # Average student count over last 5 frames

# =============================================================================
# SECTION 5: LIVE OCCUPANCY THRESHOLDS
# =============================================================================
_medium_min = 3     # Student count ≥ 3 → MEDIUM occupancy (AC ON @ 24°C)
_high_min   = 10    # Student count ≥ 10 → HIGH occupancy (AC ON @ 20°C)


def set_thresholds(medium_min: int = None, high_min: int = None):
    """
    PROCESS: Update Occupancy Classification Thresholds
    Called by server.py api_config() when user saves threshold settings.
    """
    global _medium_min, _high_min
    if medium_min is not None:
        _medium_min = max(1, int(medium_min))
    if high_min is not None:
        _high_min = max(2, int(high_min))
    print(f"[inference] Thresholds updated: MEDIUM≥{_medium_min}  HIGH≥{_high_min}")


# =============================================================================
# SECTION 6: STUDENT COUNT → OCCUPANCY BAND MAPPING
# =============================================================================
def _count_to_occupancy(student_count: int, janitor_detected: bool):
    """
    PROCESS: Occupancy Band Classification
    Rules:
      janitor only (no students) → LOW   (AC stays OFF — cleaner present)
      0 students                  → LOW   (AC OFF)
      1 to (medium_min-1)         → LOW   (too few for comfort cooling)
      medium_min to (high_min-1)  → MEDIUM (moderate cooling @ 24°C)
      high_min or more            → HIGH   (maximum cooling @ 20°C)
    Returns: (occupancy_label: str, confidence: float)
    """
    if janitor_detected and student_count == 0:
        return "low", 0.97
    if   student_count == 0:           return "low",    0.97
    elif student_count < _medium_min:  return "low",    0.93
    elif student_count < _high_min:    return "medium", 0.93
    else:                              return "high",   0.93


# =============================================================================
# SECTION 7: MAIN PREDICTION FUNCTION
# =============================================================================
def predict_frame(frame_bgr: np.ndarray) -> dict:
    """
    PROCESS: Full Frame Object Detection Pipeline
    Called by server.py background AI thread every 300ms.

    Steps:
      1. RATE LIMITING      — Return cached result if called too soon
      2. MODEL SAFETY CHECK — Return safe zeros if model failed to load
      3. RUN YOLO INFERENCE — Detect all persons in frame at 640×640
      4. PARSE DETECTIONS   — Separate into student & janitor lists
      5. SLIDING SMOOTHING  — Average student count over last N frames
      6. BAND CLASSIFICATION— Map smoothed count to LOW/MEDIUM/HIGH
      7. RETURN RESULT DICT — occupancy, bboxes, confidence, counts
    """
    global _last_call, _last_result, _vote_buf

    now = time.time()
    if now - _last_call < _interval and _last_result is not None:
        return _last_result

    if _model is None:
        return {
            "occupancy":        "low",
            "confidence":       0.0,
            "raw_scores":       {},
            "person_count":     0,
            "janitor_detected": False,
            "bboxes":           [],
            "bboxes_labeled":   [],
            "class_counts":     {}
        }

    try:
        results = _model.predict(
            source  = frame_bgr,
            conf    = CONF_THRESH,
            imgsz   = 640,
            verbose = False,
            stream  = False
        )

        student_bboxes   = []
        janitor_bboxes   = []
        janitor_detected = False

        for r in results:
            for box in r.boxes:
                cls_id         = int(box.cls[0])
                score          = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bw = x2 - x1
                bh = y2 - y1

                if cls_id == 0:
                    student_bboxes.append((x1, y1, bw, bh, "student", score))
                elif cls_id == 1:
                    janitor_bboxes.append((x1, y1, bw, bh, "janitor", score))
                    janitor_detected = True

        student_count = len(student_bboxes)

        _vote_buf.append(student_count)
        if len(_vote_buf) > VOTE_WIN * 3:
            _vote_buf.pop(0)
        smooth = round(sum(_vote_buf) / len(_vote_buf))

        occ, conf = _count_to_occupancy(smooth, janitor_detected)

        _last_call   = now
        _last_result = {
            "occupancy":        occ,
            "confidence":       conf,
            "raw_scores":       {},
            "person_count":     smooth,
            "janitor_detected": janitor_detected,
            "bboxes":           [(x, y, w, h) for x, y, w, h, *_ in student_bboxes + janitor_bboxes],
            "bboxes_labeled":   student_bboxes + janitor_bboxes,
            "class_counts":     {"student": student_count, "janitor": len(janitor_bboxes)},
        }
        print(f"[inference] students={smooth} janitor={janitor_detected} occ={occ}")
        return _last_result

    except Exception as e:
        print(f"[inference] Prediction error: {e}")
        if _last_result:
            return _last_result
        return {
            "occupancy":        "low",
            "confidence":       0.0,
            "raw_scores":       {},
            "person_count":     0,
            "janitor_detected": False,
            "bboxes":           [],
            "bboxes_labeled":   [],
            "class_counts":     {}
        }
