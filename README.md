We created our own classroom object-detection dataset with two classes: student and cleaner. We manually annotated the students and cleaners using bounding boxes and stored the annotations in YOLO detection format.

Because the original dataset was limited, we used an augmented version of the dataset for training. The training code points to:

`dataset_augmented/data.yaml`

Data augmentation means creating additional variations from the existing training images. For example, images can be flipped, resized, rotated slightly, cropped, or adjusted for brightness and contrast. These variations help the model learn how students and cleaners may appear under different classroom conditions, such as different camera angles, lighting conditions, positions, distances and image quality.

The main reason for using augmentation was to reduce overfitting. Overfitting happens when a model memorizes the training images instead of learning the general visual characteristics of students and cleaners. An overfitted model may perform well on the images it has already seen, but it may perform poorly on new classroom images or videos.

By training with an augmented dataset, the model sees more visual variations. Therefore, it is less likely to memorize only the original images and is more likely to generalize to unseen classroom footage.

Before training, we checked the dataset structure, image and label matching, class IDs and YOLO bounding-box format. The `data.yaml` file defined the training, validation and test folders and mapped class ID 0 to student and class ID 1 to cleaner.

We then used the Ultralytics YOLO training pipeline to train the custom object-detection model for 50 epochs, with an image size of 640 and a batch size of 8. The model was trained using our augmented student and cleaner dataset.

After training, YOLO generated `best.pt` and `last.pt`. We selected `best.pt` because it represents the model weights with the best validation performance during training. We then evaluated the model using precision, recall and mAP metrics and tested it on classroom images and videos to detect and separately count students and cleaners.

The notebook confirms that the augmented dataset was used for training. However, it does not show the exact augmentation methods that were applied, so we should not claim specific techniques unless they are available in the augmentation code.


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
