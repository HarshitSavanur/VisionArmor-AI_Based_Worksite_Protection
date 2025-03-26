# VisionArmor: PPE Detection for Construction Workers

## Overview
VisionArmor is a **Streamlit-based web application** that detects Personal Protective Equipment (**PPE**) in construction workers using a deep learning model trained on a construction site safety dataset.

## Live Deployment
Access the deployed application here: [VisionArmor](https://visionarmor.streamlit.app/)

## Dataset
The model is trained on the **Construction Site Safety Image Dataset** from Kaggle:
[Dataset Link](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

## Repository
GitHub Repository: [VisionArmor](https://github.com/HarshitSavanur/VisionArmor-AI_Based_Worksite_Protection.git)

## Directory Structure
```
visionarmor/
│── dataset/
│   ├── test/
│   ├── train/
│   ├── valid/
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│
│── models/
│   ├── best.pt  # Trained YOLO model for PPE detection
│
│── runs/
│   ├── detect/  # Output detection results
│
│── app.py       # Main Streamlit app
│── index.html   # Frontend UI
│── packages.txt # Dependencies
│── README.md    # This file
│── requirements.txt  # Python dependencies
│── visionarmor-ppe-prediction.ipynb  # Model training notebook
```

## Model
- The model used is **YOLO (You Only Look Once)** for object detection.
- The best-performing model is stored as **best.pt**.
- The model detects various PPE like **helmets, vests, and gloves**.

## How to Run Locally
1. Clone the repository:
   ```sh
   git clone https://github.com/HarshitSavanur/VisionArmor-AI_Based_Worksite_Protection.git
   cd visionarmor
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Future Improvements
- Improve model accuracy by using a larger dataset.
- Deploy the model on more robust cloud services.
- Add more PPE categories and worker posture detection.

