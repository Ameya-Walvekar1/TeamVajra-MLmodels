# UAV Autonomous AI Context-Switching & Mission Control System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)

**Team Vajra — Multi-Modal Aerial Reconnaissance & Autonomous Disaster Response ML Engine**

An end-to-end mission control console designed for Unmanned Aerial Vehicles (UAVs). The system features **real-time inference-driven dynamic context switching** across domain-specific deep learning models for human search & rescue, fire/smoke hazard analysis, thermal infrared detection, and flood terrain segmentation with dynamic D* Lite path planning.

---

## Key Features

- **Inference-Driven Dynamic Context Switching**: Autonomously infers operational context from incoming imagery or video streams (spectral variance, flame/smoke ratios, thermal IR signatures, inundation coverage) and switches active models dynamically without fixed timers.
- **Multi-Modal Model Pipeline**:
  - **Visible-Spectrum Human Detection**: High-precision daylight aerial search and rescue.
  - **Fire & Smoke Hazard Detection**: Real-time identification of flame fronts and smoke plumes.
  - **Thermal Infrared Search**: Low-visibility human search using infrared radiometric spectrums.
  - **FloodNet Semantic Segmentation & D\* Lite Path Planning**: 10-class disaster terrain segmentation, traversability cost mapping, and dynamic obstacle replanning.
- **Real-Time Video Stream Telemetry**: Live frame-by-frame inference preview, context switching timeline event recording, and coordinate export.
- **Mission Control Dashboard**: Professional light-theme UI built with Streamlit and CSS custom design tokens.

---

## Model Registry & Architecture

The system coordinates four domain-specific inference models managed by a central scene context classifier:

| System Model Label | Backend Registry ID | Operational Context Target | Model Architecture / Base | Weight File Path |
|---|---|---|---|---|
| **Model 1** | `model1` | `Visible Person Search` | YOLOv8 Daylight Reconnaissance | `models/person/best.pt` |
| **Model 2** | `model2` | `Fire / Smoke` | YOLOv8 Hazard Detection | `models/fire_smoke/best.pt` |
| **Model 3** | `model4` | `Terrain / Navigation` | LRASPP MobileNetV3 FloodNet (512x512) | `models/terrain/best_floodnet.pth` |
| **Model 4** | `model5` | `Thermal Search` | YOLOv8 Infrared Thermal | `models/thermal/best.pt` |
| **Model 5 / Main** | `model6` | `Inference-Driven Context Switching` | Autonomous Scene Classifier | `models/classifier/best.pt` |

> **Registry Mapping Note**: Internal backend IDs (`model1`, `model2`, `model4`, `model5`, `model6`) map directly to human-readable system labels **Model 1** through **Model 5 / Main**.

---

## Installation & Setup Guide

### 1. Prerequisites
Ensure the system has Python 3.10 or higher installed along with Git. CUDA-capable GPU hardware is automatically detected if available.

### 2. Clone the Repository
```bash
git clone https://github.com/Ameya-Walvekar1/TeamVajra-MLmodels.git
cd TeamVajra-MLmodels
```

### 3. Create & Activate Virtual Environment

- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Windows (Command Prompt / PowerShell)**:
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution Instructions

### Option A: Using Helper Launch Scripts

- **Linux / macOS**:
  ```bash
  chmod +x run_linux.sh
  ./run_linux.sh
  ```

- **Windows**:
  ```cmd
  run_windows.bat
  ```

### Option B: Direct Streamlit Command
```bash
streamlit run app/gui.py
```

Once launched, open your web browser and navigate to:
```
http://localhost:8501
```

---

## Directory Structure

```
TeamVajra-MLmodels/
├── README.md                   <-- Technical documentation & guide
├── requirements.txt            <-- Project dependencies
├── run_linux.sh                <-- Linux environment launcher
├── run_windows.bat             <-- Windows launcher
├── .streamlit/
│   └── config.toml             <-- UI theme configuration
├── app/
│   ├── gui.py                  <-- Streamlit Mission Control application entry point
│   ├── styles.py               <-- Engineering light theme styling
│   └── video_processor.py      <-- Video stream processing & timeline builder
├── config/
│   └── model_registry.py       <-- Central model metadata & path resolution
├── fusion/                     <-- Multi-sensor telemetry fusion modules
├── inference/
│   ├── cost_map.py             <-- FloodNet terrain cost map generator
│   ├── dstar_lite.py           <-- D* Lite dynamic path planning algorithm
│   ├── dstar_lite_engine.py    <-- Real-time replanning pipeline
│   ├── model4_engine.py        <-- FloodNet semantic segmentation engine
│   ├── model4_pipeline.py      <-- Integrated terrain & navigation pipeline
│   └── yolo_loader.py          <-- Unified YOLO model loader & bounding box drawer
├── models/                     <-- Model weights directory
│   ├── classifier/             <-- Model 5 Main Switcher weights (best.pt) [model6]
│   ├── fire_smoke/             <-- Model 2 Fire & Smoke weights (best.pt) [model2]
│   ├── person/                 <-- Model 1 Visible Person weights (best.pt) [model1]
│   ├── terrain/                <-- Model 3 Terrain weights (best_floodnet.pth) [model4]
│   └── thermal/                <-- Model 4 Thermal Search weights (best.pt) [model5]
├── scripts/                    <-- Helper deployment utilities
├── switching/
│   ├── model_manager.py        <-- Model lifecycle manager & hardware metrics
│   └── model_switcher.py       <-- Inference classifier & context switching engine
└── tests/
    └── test_pipeline.py        <-- Automated system verification test suite
```

---

## Verification & Automated Testing

The codebase includes an automated test suite verifying model registry integrity, inference loader pipelines, context switching logic, and video processing handlers.

To execute the test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Attribution & Project Info

Developed exclusively by **Team Vajra** for UAV Autonomous Search & Rescue Missions.
