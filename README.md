# 🌿 PlantGuard AI — Plant Disease Detection System

An AI-powered agricultural diagnosis platform that classifies plant diseases from leaf imagery using deep learning (transfer learning with **EfficientNet-B0**). The system features an interactive single-page web dashboard, a Flask REST API backend, sample datasets, and full project documentation.

---

## 📌 Project Overview

Early diagnosis of plant diseases is critical for reducing crop yield loss. PlantGuard AI provides instant identification, severity assessment, and actionable treatment recommendations.

* **Primary Crops:** Tomato & Brinjal (Eggplant)
* **Classes:** 20 distinct disease and healthy states
* **Architecture:** Transfer Learning on EfficientNet-B0 (PyTorch)
* **Performance:** **98.44%** Validation Accuracy
* **Checkpoint Size:** ~17 MB (`models/plant_disease_model.pth`)

---

## 🗂️ Supported Classes (20 Classes)

| Crop | Classes / Diseases |
| :--- | :--- |
| **Brinjal (Eggplant)** | Aphids, Cercospora Leaf Spot, Defect Eggplant, Flea Beetles, Fresh Eggplant, Fresh Eggplant Leaf (Healthy), Leaf Wilt, Phytophthora Blight, Powdery Mildew, Tobacco Mosaic Virus |
| **Tomato** | Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted spider mite), Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus |

---

## 🚀 Key Features

* **Real-Time Web Interface (`plant_disease_website.html`):** Drag-and-drop leaf image upload, live preview, progress simulation, and top-5 probability breakdown.
* **Diagnosis & Advisory:** Displays disease severity (None, Medium, High, Critical), disease symptoms, and organic/chemical treatment advice.
* **Lightweight REST API (`plant_disease_backend.py`):** Flask inference server exposing endpoints for prediction, health check, class registry, and model metadata.
* **End-to-End Pipeline:** Colab training notebook (`plant_disease_training_colab.ipynb`) for training or fine-tuning from scratch.
* **Project Documentation (`CSPfinal.docx`):** Comprehensive project report covering problem statement, system architecture, methodology, literature review, and experimental results.

---

## 📂 Project Structure

```text
├── models/
│   └── plant_disease_model.pth          # Fine-tuned PyTorch EfficientNet-B0 weights (~17 MB)
├── samples/                             # Sample leaf test images & manifest
│   ├── manifest.json
│   └── *.jpg
├── plant_disease_backend.py             # Flask API server & PyTorch inference engine
├── plant_disease_website.html           # Single-page web application UI
├── plant_disease_training_colab.ipynb   # Google Colab notebook for model training
├── CSPfinal.docx                        # Complete project report documentation
├── .gitignore                           # Git ignore rules for virtual environments & build artifacts
└── README.md                            # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.9+ installed
* PyTorch (`torch`, `torchvision`)
* Flask, Pillow

### 2. Clone Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### 3. Install Dependencies
```bash
pip install flask pillow torch torchvision
```

*(For GPU support on Windows/Linux, install PyTorch with your CUDA version from [pytorch.org](https://pytorch.org)).*

---

## 💻 Running the Application

### Start the Flask Server
```bash
python plant_disease_backend.py
```

By default, the server runs on:
👉 **`http://127.0.0.1:5000`**

### Access the Web App
Open your web browser and go to `http://127.0.0.1:5000`. Navigate to the **Scan** tab, upload a sample leaf photo from `samples/`, and click **Analyze Plant Disease**.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main web dashboard |
| `GET` | `/api/health` | Backend status, device (CPU/GPU), and model loading state |
| `GET` | `/api/classes` | Returns all 20 supported disease classes & treatment info |
| `GET` | `/api/model` | Model metadata, validation accuracy, and training history |
| `POST` | `/api/predict` | Multipart form-data with `image` file; returns top-5 prediction JSON |

### Sample Prediction Request
```bash
curl -X POST -F "image=@samples/Late_blight_on_tomato_leaf_(7871756748).jpg" http://127.0.0.1:5000/api/predict
```

---

## 🎓 Academic / Project Report

The repository includes the complete project report in [`CSPfinal.docx`](./CSPfinal.docx), detailing:
1. Introduction & Agricultural Significance
2. Literature Survey on Convolutional Neural Networks & Plant Pathology
3. Proposed Deep Learning Architecture (EfficientNet-B0)
4. Data Preprocessing & Augmentation Pipelines
5. Training Results, Confusion Matrix, and Performance Evaluation
6. Web Deployment and REST Integration

---

## 📄 License
This project is developed for academic and educational research.

