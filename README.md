# VayuV — UAV Autonomous AI Context-Switching & Mission Control System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)

**VayuV — Multi-Modal Aerial Reconnaissance & Autonomous Disaster Response ML Engine**

VayuV is an end-to-end mission control and AI inference system designed for **Unmanned Aerial Vehicles (UAVs)** operating in search-and-rescue and disaster-response environments.

The system enables **real-time, inference-driven context switching** between domain-specific deep learning models for human detection, fire and smoke analysis, thermal search, and flood-terrain understanding. Based on the detected operational context, VayuV dynamically activates the appropriate AI model and, where required, integrates terrain analysis with **D* Lite dynamic path planning**.

---

## Key Features

### 🧠 Inference-Driven Dynamic Context Switching

VayuV continuously analyzes incoming visual data and determines the current operational context.

Instead of relying on fixed timers or manually selected models, the system uses scene-level inference signals such as:

* Spectral and visual characteristics
* Flame and smoke presence
* Thermal infrared signatures
* Flood/inundation coverage
* Terrain characteristics

Based on the inferred context, VayuV dynamically switches between specialized AI models.

### 🤖 Multi-Modal AI Model Pipeline

VayuV integrates multiple domain-specific models:

* **Visible-Spectrum Human Detection**

  * Daylight aerial search and rescue
  * Detects people from UAV imagery

* **Fire & Smoke Hazard Detection**

  * Detects flames and smoke
  * Supports hazard identification during disaster reconnaissance

* **Thermal Infrared Search**

  * Searches for humans using thermal imagery
  * Designed for low-visibility and nighttime conditions

* **FloodNet Semantic Segmentation**

  * Performs terrain and flood-region segmentation
  * Generates terrain traversability information

* **D* Lite Dynamic Path Planning**

  * Converts terrain information into navigation costs
  * Performs dynamic path planning
  * Supports replanning when environmental conditions change

### 📡 Real-Time Video & Telemetry

The system provides a mission-control interface capable of:

* Live video processing
* Frame-by-frame AI inference
* Real-time model switching
* Context-switching event logging
* Detection visualization
* Coordinate/event export
* Mission status monitoring

### 🖥️ Mission Control Dashboard

VayuV includes a professional **Streamlit-based mission control dashboard** with a custom engineering-focused light theme.

The dashboard provides visibility into:

* Active AI model
* Current operational context
* Detection results
* Model-switching events
* Video inference
* Navigation information
* System status

---

# Model Registry & Architecture

VayuV coordinates multiple domain-specific inference models through a central context-classification and model-switching pipeline.

| System Model Label | Backend Registry ID | Operational Context                | Model Architecture / Base             | Weight File Path                   |
| ------------------ | ------------------- | ---------------------------------- | ------------------------------------- | ---------------------------------- |
| **Model 1**        | `model1`            | Visible Person Search              | YOLOv8 Daylight Reconnaissance        | `models/person/best.pt`            |
| **Model 2**        | `model2`            | Fire / Smoke                       | YOLOv8 Hazard Detection               | `models/fire_smoke/best.pt`        |
| **Model 3**        | `model4`            | Terrain / Navigation               | LRASPP MobileNetV3 FloodNet (512×512) | `models/terrain/best_floodnet.pth` |
| **Model 4**        | `model5`            | Thermal Search                     | YOLOv8 Infrared Thermal               | `models/thermal/best.pt`           |
| **Model 5 / Main** | `model6`            | Inference-Driven Context Switching | Autonomous Scene Classifier           | `models/classifier/best.pt`        |

> **Registry Mapping Note:** Internal backend IDs (`model1`, `model2`, `model4`, `model5`, `model6`) map directly to the human-readable system labels **Model 1** through **Model 5 / Main**.

---

# System Architecture

At a high level, VayuV follows the pipeline:

```text
                UAV Sensors / Camera
                        │
                        ▼
                ┌───────────────┐
                │ Video / Frame │
                │    Input      │
                └───────┬───────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ Context Classifier  │
              │      Model 5        │
              └──────────┬──────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
     Person Search   Fire / Smoke   Thermal Search
       Model 1          Model 2        Model 4
          │              │               │
          └──────────────┼───────────────┘
                         │
                         ▼
                Terrain / Navigation
                      Model 3
                         │
                         ▼
                 Terrain Cost Map
                         │
                         ▼
                  D* Lite Planner
                         │
                         ▼
                Dynamic UAV Path
```

The architecture is designed to separate:

* **Context understanding**
* **Domain-specific perception**
* **Terrain understanding**
* **Navigation and replanning**
* **Mission-control visualization**

---

# Installation & Setup

## 1. Prerequisites

Ensure the system has:

* Python 3.10 or higher
* Git
* CUDA-capable GPU (optional, but recommended for accelerated inference)

VayuV automatically uses available GPU hardware where supported.

---

## 2. Clone the Repository

```bash
git clone https://github.com/Ameya-Walvekar1/VayuV.git
cd VayuV
```

> If the GitHub repository URL has a different exact casing or path, replace the URL above with the repository's actual URL.

---

## 3. Create & Activate Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows — Command Prompt / PowerShell

```cmd
python -m venv venv
.\venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Execution

## Option A — Using Helper Launch Scripts

### Linux / macOS

```bash
chmod +x run_linux.sh
./run_linux.sh
```

### Windows

```cmd
run_windows.bat
```

---

## Option B — Direct Streamlit Command

```bash
streamlit run app/gui.py
```

Once the application starts, open:

```text
http://localhost:8501
```

in your browser.

---

# Directory Structure

```text
VayuV/
│
├── README.md
├── requirements.txt
├── run_linux.sh
├── run_windows.bat
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── gui.py
│   ├── styles.py
│   └── video_processor.py
│
├── config/
│   └── model_registry.py
│
├── fusion/
│   └── ...
│
├── inference/
│   ├── cost_map.py
│   ├── dstar_lite.py
│   ├── dstar_lite_engine.py
│   ├── model4_engine.py
│   ├── model4_pipeline.py
│   └── yolo_loader.py
│
├── models/
│   ├── classifier/
│   │   └── best.pt
│   │
│   ├── fire_smoke/
│   │   └── best.pt
│   │
│   ├── person/
│   │   └── best.pt
│   │
│   ├── terrain/
│   │   └── best_floodnet.pth
│   │
│   └── thermal/
│       └── best.pt
│
├── scripts/
│   └── ...
│
├── switching/
│   ├── model_manager.py
│   └── model_switcher.py
│
└── tests/
    └── test_pipeline.py
```

---

# Core Components

### `app/`

Contains the VayuV mission-control interface and video-processing logic.

* `gui.py` — Streamlit mission-control application
* `styles.py` — UI styling and design tokens
* `video_processor.py` — video inference and context timeline processing

### `config/`

Central configuration and model metadata.

* `model_registry.py` — model definitions, paths, and metadata

### `inference/`

Contains the individual inference and navigation pipelines.

* `cost_map.py` — generates terrain traversal costs
* `dstar_lite.py` — D* Lite path-planning implementation
* `dstar_lite_engine.py` — real-time dynamic replanning
* `model4_engine.py` — FloodNet semantic segmentation engine
* `model4_pipeline.py` — integrated terrain/navigation pipeline
* `yolo_loader.py` — unified YOLO loading and visualization

### `switching/`

Responsible for AI model lifecycle management and dynamic context switching.

* `model_manager.py` — model loading, lifecycle, and hardware metrics
* `model_switcher.py` — context classification and model-switching logic

### `fusion/`

Contains modules responsible for multi-sensor and telemetry fusion.

### `tests/`

Automated tests for verifying the VayuV inference and switching pipeline.

---

# Context-Switching Workflow

VayuV's model-selection pipeline follows the general workflow:

```text
Input Frame
     │
     ▼
Context Analysis
     │
     ▼
Operational Context
     │
     ├── Visible Person Search ──► Model 1
     │
     ├── Fire / Smoke ───────────► Model 2
     │
     ├── Terrain / Flood ────────► Model 3
     │
     └── Thermal Search ─────────► Model 4
```

The context classifier acts as the central decision layer while the domain-specific models perform specialized inference.

---

# Terrain & Navigation Pipeline

For terrain-aware navigation, VayuV combines semantic segmentation with dynamic path planning.

```text
UAV Image
    │
    ▼
FloodNet Segmentation
    │
    ▼
Semantic Terrain Classes
    │
    ▼
Traversability Cost Map
    │
    ▼
D* Lite
    │
    ▼
Optimal Available Path
    │
    ▼
UAV Navigation
```

When the environment changes or previously available paths become obstructed, the D* Lite planner can update the path using the modified cost information.

---

# Verification & Automated Testing

The repository includes an automated test suite covering:

* Model registry integrity
* Model-loading pipelines
* Context-switching logic
* Video-processing handlers
* Inference pipeline components

Run the complete test suite using:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

# Technology Stack

| Component             | Technology           |
| --------------------- | -------------------- |
| Programming Language  | Python 3.10+         |
| Deep Learning         | PyTorch              |
| Object Detection      | YOLOv8               |
| Semantic Segmentation | LRASPP + MobileNetV3 |
| Terrain Dataset       | FloodNet             |
| Path Planning         | D* Lite              |
| Mission Dashboard     | Streamlit            |
| Video Processing      | OpenCV               |
| GPU Acceleration      | CUDA                 |
| UI Styling            | Custom CSS           |

---

# Intended Applications

VayuV is designed around autonomous aerial reconnaissance and disaster-response scenarios including:

* Search and rescue
* Fire and smoke reconnaissance
* Thermal human detection
* Flood assessment
* Terrain-aware UAV navigation
* Dynamic obstacle-aware mission planning
* Multi-context aerial surveillance

---

# Project

**VayuV** is developed by **Team Vajra** for UAV-based autonomous search, rescue, reconnaissance, and disaster-response missions.

The system combines computer vision, deep learning, sensor fusion, terrain understanding, and autonomous navigation into a unified UAV AI mission-control architecture.

---

## Attribution

Developed exclusively by **Team Vajra**.

**Project:** VayuV
**Domain:** UAV • AI • Computer Vision • Autonomous Navigation • Disaster Response
