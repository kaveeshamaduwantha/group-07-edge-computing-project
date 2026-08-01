# 🏫 Smart Classroom Edge AI — Energy Efficiency & Occupancy Control

> **Group 07 — Edge Computing Project**  
> An end-to-end Edge AI solution integrating real-time computer vision (YOLOv8) with a smart HVAC thermal simulation engine to optimize classroom energy consumption.

---

## 🌟 Key Features

- 📹 **Live Video Stream & Detection:** Real-time YOLOv8 object detection distinguishing `student` and `janitor` classes.
- ❄️ **Dynamic AC Climate Control:** Automatically adjusts target AC cooling temperatures based on classroom occupancy bands (LOW, MEDIUM, HIGH).
- 🧹 **Janitor Cleaning Protection:** Immediately turns OFF the AC system for 15 seconds when a janitor is detected, extending shutdown by +15s if cleaning continues (background check every 5s).
- 🌡️ **Thermal Model Simulation:** Real-time room temperature decay and heat dissipation physics simulation.
- 🐳 **Dockerized Edge Deployment:** Fully containerized microservice ready for deployment on edge gateway devices.

---

## 👥 Team & Roles

| Role | Name / Contributor | Responsibilities |
|---|---|---|
| **Product Owner** | Team Lead | Product Backlog, User Stories & Acceptance Criteria |
| **Scrum Master** | Agile Coordinator | Sprint Planning, GitHub Projects Kanban Board & Process |
| **Data Scientist 1** | AI Model Lead | YOLOv8 Custom Architecture Training & Dataset Annotation |
| **Data Scientist 2** | Computer Vision Engineer| Model Evaluation, Precision/Recall Metrics & NMS Tuning |
| **Data Scientist 3** | Dataset Specialist | Janitor vs. Student Edge Case Feature Extraction |
| **Software Developer 1**| Backend Engineer | Flask Edge Streaming API & Async Threading Engine |
| **Software Developer 2**| Simulation Engineer | AC Control Logic, Janitor Countdown & Thermal Physics |
| **Software Developer 3**| Full-Stack & DevOps | UI Dashboard, Real-time Visuals, Docker & Containerization |

---

## 🏗️ System Architecture

```
+---------------------+     +-----------------------+     +------------------------+
|  Camera / Video     | --> |  YOLOv8 AI Engine     | --> |  AC Simulation Engine  |
|  (RTSP / MP4 Feed)  |     |  (Student / Janitor)  |     |  (Occupancy & Temp)    |
+---------------------+     +-----------------------+     +------------------------+
                                                                      |
                                                                      v
                                                         +------------------------+
                                                         |  Web Dashboard (Flask) |
                                                         |  (http://localhost:5000)|
                                                         +------------------------+
```

---

## 🚀 Quick Start Guide

### Option 1: Running with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/kaveeshamaduwantha/group-07-edge-computing-project.git
cd group-07-edge-computing-project

# Build and start container
docker-compose up --build
```
Access the live dashboard at: **`http://localhost:5000`**

### Option 2: Running Locally with Python

```bash
# Install dependencies
pip install -r edge_app/requirements.txt

# Start the Flask edge server
python edge_app/server.py
```

---

## 📊 Occupancy Rules & Temperature Logic

| Occupancy Level | Student Count | AC State | Target Temp | Fan Speed |
|---|---|---|---|---|
| **OFF (Janitor)** | Janitor Detected | **OFF** | -- | 0 Fans |
| **LOW** | 0 – 2 Students | **OFF** | -- | 0 Fans |
| **MEDIUM** | 3 – 9 Students | **ON** | 24°C *(Configurable)* | 2 Fans |
| **HIGH** | 10+ Students | **ON** | 20°C *(Configurable)* | 4 Fans |

---

## 🛠️ Technology Stack

- **Computer Vision:** YOLOv8, PyTorch, OpenCV
- **Backend:** Python 3.11, Flask, Flask-CORS
- **Frontend:** Vanilla JS, HTML5, CSS3 (Glassmorphism), Chart.js
- **DevOps:** Docker, Docker Compose, Linux Debian Slim
