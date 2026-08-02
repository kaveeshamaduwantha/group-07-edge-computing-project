# Student/Cleaner Object Detection Project

This repository contains a data-science and machine-learning workflow for training a YOLO object-detection model to detect two classes in CCTV-style images:

- 0 `student`
- 1 `cleaner`

The project is designed for collaboration between data scientists, developers, and application teams.

## Project Overview

The goal of this project is to build a reliable detection system that can support tasks such as:

- locating students in video frames,
- identifying cleaning staff,
- supporting downstream analytics for monitoring and reporting.

## Repository Purpose

This repository is organized to support:

- dataset preparation and annotation workflows,
- model training and evaluation,
- export of trained weights for application integration,
- collaboration through GitHub.

## Project Structure

```text
myDataSet/
  dataset_augmented/
  predictions/
  training_results/
    classroom_test_run/
      weights/
        best.pt
  validation_results/
src/
README.md
```

## Data Science Setup

Use the following steps to prepare the environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Trained Model

The best trained model weights are available at:

```text
training_results/classroom_test_run/weights/best.pt
```

This file is the main artifact used by application developers for inference.

## How App Developers Can Use the Model

App developers can load the trained model directly for image or video inference.

### Example: Python inference

```python
from ultralytics import YOLO

model = YOLO("training_results/classroom_test_run/weights/best.pt")

results = model("path/to/image.jpg", conf=0.25)

for result in results:
    boxes = result.boxes
    for box in boxes:
        print(box.cls, box.conf)
```

### Notes for integration

- Use a confidence threshold such as `0.25` for initial testing.
- The model is expected to return predictions for the `student` and `cleaner` classes.
- For production deployment, validate the model on real-world images and CCTV footage before full rollout.

## Dataset Summary

The current dataset includes:

- 960 unique images
- 672 training images
- 192 validation images
- 96 test images
- 6,668 `student` annotations
- 76 `cleaner` annotations

## Why Data Augmentation Was Used

Data augmentation was applied because the original dataset was limited in size and did not cover all real-world conditions that may appear in CCTV footage. Without augmentation, the model may overfit to training images and perform poorly on new scenes.

Augmentation helps improve generalization by simulating changes in:

- lighting,
- camera angle,
- image quality,
- motion blur,
- shadows,
- object position.

## GitHub for Developers

This repository is intended for collaborative development. The following workflow is recommended:

1. Create a new branch:
   ```bash
   git checkout -b feature/your-task-name
   ```
2. Make your changes locally.
3. Commit clearly:
   ```bash
   git add .
   git commit -m "Describe your change clearly"
   ```
4. Push the branch:
   ```bash
   git push origin feature/your-task-name
   ```
5. Open a pull request for review.

### Recommended branch names

- `feature/description`
- `bugfix/description`
- `docs/description`
- `experiment/description`

## Contributors

### Data Annotation Team

The following contributors supported annotation and review work:

- B.V. Amasha Chathumini
- S.T.K. Maduwantha Thewarapperuma
- Tharmapala Muralitharan

### Group Support

All group members contributed by supporting data collection, feedback, and project coordination.
