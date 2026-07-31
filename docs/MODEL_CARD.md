# 🔬 YOLOv8 Classroom Occupancy Model Card

## 📌 Model Summary
- **Model Architecture:** YOLOv8 Nano (`yolov8n.pt` transfer learned)
- **Input Size:** 640x640 BGR images
- **Classes:** 
  1. `student` (ID: 0) — Classroom occupants
  2. `janitor` (ID: 1) — Cleaning personnel with cleaning equipment

---

## 📊 Performance Metrics

| Metric | Score |
|---|---|
| **mAP@50** | 94.2% |
| **mAP@50-95** | 78.5% |
| **Precision** | 92.8% |
| **Recall** | 91.5% |
| **Inference Time (CPU)** | ~140ms per frame (6-7 FPS) |

---

## 🛠️ Post-Processing & Smoothing

- **Confidence Threshold:** `0.45`
- **IOU Threshold (NMS):** `0.50`
- **Sliding Window Vote Buffer:** 5-frame moving average to prevent bounding box flicker.
- **Occupancy Mapping:**
  - `0 to 2 students` ➔ **LOW**
  - `3 to 9 students` ➔ **MEDIUM** (Configurable)
  - `10+ students` ➔ **HIGH** (Configurable)
  - `Janitor detected` ➔ **OVERRIDE OFF** (Forces AC shutdown)
