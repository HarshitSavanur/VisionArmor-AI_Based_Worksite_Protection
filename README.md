# PPE Detection for Construction Workers

## Introduction
Construction sites are hazardous environments, often resulting in accidents due to the lack of proper safety equipment. This project aims to detect Personal Protective Equipment (PPE) on workers, which can be further utilized for tracking and triggering alarms for safety monitoring.

We utilized the **Construction Site Safety Image Dataset** provided by Roboflow. The dataset consists of **2,801 images** labeled in the YOLOv8 format, split into:
- **Train**: 2,605 images
- **Validation**: 114 images
- **Test**: 82 images

### Classes Detected:
The model is trained to detect the following 10 classes:
- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- Machinery
- Vehicle

---

## Setup
The model training and inference were conducted on **Kaggle** using a **P100 GPU**. We utilized the **Ultralytics YOLOv8** library for custom object detection.

To install the necessary dependencies, run:
```bash
pip install ultralytics streamlit opencv-python
```

---

---

## Streamlit Application
We built a **Streamlit** web application for easy usage.

### Features:
- **Video Processing:** Upload videos and detect PPE in real-time.
- **Image Processing:** Upload images and identify PPE.
- **Safety Blog:** Informative content on construction site safety.

To run the application:
```bash
streamlit run app.py
```

---

## Model Training & Inference
### Training:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data/dataset.yaml", epochs=100, imgsz=640)
```

### Inference:
```python
results = model.predict(source="source_files/test_images", save=True)
```

---

## Future Enhancements
- **Live CCTV Integration**: Real-time PPE monitoring on construction sites.
- **Automated Alerts**: Trigger alarms when a worker lacks necessary safety gear.
- **Deployment on Edge Devices**: Optimizing for deployment on low-power edge devices.


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

