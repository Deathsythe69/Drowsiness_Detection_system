# 👁️ Real-Time Drowsiness & Attention Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00C0A3.svg)](https://developers.google.com/mediapipe)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52.svg?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device-green.svg)](#privacy--security)

An intelligent, privacy-first desktop application that monitors a user's facial landmarks in real time via standard webcam feeds. It continuously assesses eye closure, blink dynamics, and yawning frequency to detect fatigue, drowsiness, and sustained inattention before productivity or safety is compromised.

---

<!-- ========================================================================= -->
<!-- IMAGE PLACEHOLDER: PROJECT BANNER / HERO IMAGE                            -->
<!-- ========================================================================= -->
<p align="center">
  <img src="../assets/screenshots/banner.png" alt="Project Banner / Hero Demonstration" width="90%" />
</p>

---

## 📑 Table of Contents

- [Overview & Motivation](#-overview--motivation)
- [Key Features](#-key-features)
- [Visual Walkthrough & Screenshots](#-visual-walkthrough--screenshots)
- [Architecture & Detection Pipeline](#-architecture--detection-pipeline)
- [Mathematical Foundations](#-mathematical-foundations)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [Configuration Guide (`config.yaml`)](#-configuration-guide-configyaml)
- [Session Analytics & Local Storage](#-session-analytics--local-storage)
- [Privacy & Security](#-privacy--security)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Roadmap](#-roadmap)
- [License & Authors](#-license--authors)

---

## 🌟 Overview & Motivation

Extended screen time during study sessions, software development, online exams, and remote work often leads to fatigue-induced micro-sleeps and reduced cognitive attention. 

This system provides an **offline, edge-computed solution** designed specifically for:
- 🎓 **Students & E-Learners:** Track study session alertness and avoid fatigue burnout.
- 💻 **Remote Professionals:** Monitor focus during extended work blocks with customizable break triggers.
- 📝 **Exam Proctoring & Workstation Supervision:** Detect inattention or prolonged absence without transmitting intrusive video streams.

---

## ✨ Key Features

- ⚡ **Real-Time Landmark Tracking:** Powered by MediaPipe 468-point Face Mesh running smoothly at ≥20 FPS on standard CPUs.
- 👁️ **Eye Aspect Ratio (EAR) Analysis:** Robust metric for detecting micro-sleeps and sustained eye closure with temporal smoothing.
- 🥱 **Yawn Detection (MAR):** Computes Mouth Aspect Ratio to log yawning events as secondary fatigue indicators.
- 🎯 **Dynamic Baseline Calibration:** 10–15 second initial session calibration tailored to individual eye geometry, ambient lighting, and glasses wearers.
- 🧮 **Cognitive Math Puzzle Alarm Dismissal:** Requires solving an interactive arithmetic puzzle to dismiss auditory alerts, ensuring actual wakefulness.
- 📊 **Post-Session Analytics & Visual Summaries:** Embedded matplotlib charts depicting session EAR trends, fatigue transitions, and event breakdowns.
- 🔒 **Privacy-by-Design:** Zero raw video frames or photos are saved to disk or network. Only derived numerical telemetry is saved in local SQLite.

---

## 📸 Visual Walkthrough & Screenshots

> **Note:** Upload your application screenshots into the `assets/screenshots/` folder to populate these views.

### 1. Live Monitoring Interface
*Webcam stream with real-time landmark meshes, EAR/MAR gauges, session timer, and dynamic state indicator.*

<!-- IMAGE PLACEHOLDER: LIVE MONITOR -->
<p align="center">
  <img src="../assets/screenshots/live_monitor.png" alt="Live Monitor Interface" width="85%" />
  <br />
  <em>Figure 1: Main live monitor screen with real-time facial mesh and telemetry HUD.</em>
</p>

---

### 2. Drowsiness Warning & Alert Trigger
*Visual cues (yellow/red boundary overlays), synthesized audio alarms, and state elevation.*

<!-- IMAGE PLACEHOLDER: ALERT SCREEN -->
<p align="center">
  <img src="../assets/screenshots/alert_state.png" alt="Drowsiness Alert Triggered" width="85%" />
  <br />
  <em>Figure 2: Active alert state with overlay warning and cognitive verification trigger.</em>
</p>

---

### 3. Cognitive Math Puzzle Wake-Up Modal
*Ensures the user is genuinely alert before silencing alarms and returning to active monitoring.*

<!-- IMAGE PLACEHOLDER: MATH PUZZLE -->
<p align="center">
  <img src="../assets/screenshots/math_puzzle.png" alt="Cognitive Math Puzzle Modal" width="70%" />
  <br />
  <em>Figure 3: Interactive arithmetic puzzle required to dismiss auditory alarms.</em>
</p>

---

### 4. Settings & Sensitivity Calibration
*Customizable EAR/MAR thresholds, consecutive frame counts, camera device selection, and volume controls.*

<!-- IMAGE PLACEHOLDER: SETTINGS PANEL -->
<p align="center">
  <img src="../assets/screenshots/settings_panel.png" alt="Settings and Configuration Panel" width="80%" />
  <br />
  <em>Figure 4: Configuration window with threshold sliders and hardware device management.</em>
</p>

---

### 5. Post-Session Analytics Dashboard
*Comprehensive historical charts showing fatigue spikes, attention curves, and session summary metrics.*

<!-- IMAGE PLACEHOLDER: SESSION SUMMARY -->
<p align="center">
  <img src="../assets/screenshots/session_summary.png" alt="Session Summary and Analytics" width="85%" />
  <br />
  <em>Figure 5: Post-session analytics dashboard displaying EAR distribution over time.</em>
</p>

---

## 🏗️ Architecture & Detection Pipeline

The system uses a **multi-threaded architecture** decoupling camera acquisition and CPU-heavy landmark processing from the PyQt6 main UI loop:

```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Webcam Stream  │──────▶│ Capture Thread (Queue) │──────▶│ MediaPipe Face Mesh  │
└─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                      │
                                                                      ▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  PyQt6 UI Loop  │◀──────│  FSM Classifier &      │◀──────│ Feature Extractor    │
│  (Signals/Slots)│       │  Fatigue Accumulator   │       │ (EAR, MAR Calculation│
└────────┬────────┘       └───────────┬────────────┘       └──────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐       ┌────────────────────────┐
│ Audio & Visual  │       │ Local SQLite Storage   │
│ Alert Triggers  │       │ (Sessions & Events)    │
└─────────────────┘       └────────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Eye Aspect Ratio (EAR)
For each eye with 6 landmark coordinates $(p_1, p_2, p_3, p_4, p_5, p_6)$:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

$$\text{EAR}_{\text{avg}} = \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2}$$

- **Awake state:** Typical EAR $\approx 0.25 - 0.35$.
- **Closed / Drowsy state:** EAR drops below threshold ($\approx 0.21$) for $\ge N$ consecutive frames.

---

### 2. Mouth Aspect Ratio (MAR) - Yawn Detection
For mouth landmark coordinates $(p_1 \dots p_8)$:

$$\text{MAR} = \frac{\|p_2 - p_8\| + \|p_3 - p_7\| + \|p_4 - p_6\|}{2 \cdot \|p_1 - p_5\|}$$

- **Normal state:** Typical MAR $< 0.50$.
- **Yawn state:** MAR $> 0.60$ for $\ge 15$ consecutive frames.

---

### 3. Fatigue Scoring & Finite State Machine (FSM)
States: `AWAKE` $\rightarrow$ `DROWSY_WARNING` $\rightarrow$ `DROWSY_ALERT`

Transitions incorporate rolling-window fatigue scores, yawn occurrences, and temporal smoothing to eliminate false positives from normal blinking.

---

## 📂 Project Directory Structure

```
Drowsiness_Detection_system/
├── main.py                     # Application entry point & Qt launcher
├── config.yaml                 # Tunable thresholds, camera, and alert settings
├── requirements.txt            # Python dependencies
├── assets/                     # Sound files, icons, and screenshot assets
│   ├── alarm.wav               # Auto-synthesized or custom alert sound
│   └── screenshots/            # Place application screenshot images here
├── core/                       # Core CV and detection logic
│   ├── capture.py              # Threaded webcam frame grabber
│   ├── landmarks.py            # MediaPipe Face Mesh wrapper
│   ├── features.py             # EAR, MAR, and distance algorithms
│   ├── classifier.py           # Finite State Machine & fatigue scoring
│   └── alerts.py               # Sound synthesizer & alarm dispatcher
├── storage/                    # Local database layer
│   ├── db.py                   # SQLite connection, migrations, and queries
│   └── models.py               # Session and Event data models
├── ui/                         # PyQt6 modern dark-themed GUI
│   ├── main_window.py          # Primary monitoring screen & HUD
│   ├── math_puzzle_window.py   # Cognitive alarm verification puzzle
│   ├── settings_window.py      # Thresholds & preferences editor
│   └── summary_window.py       # Session analytics & matplotlib graphs
├── tests/                      # Automated test suite
│   ├── test_features.py        # EAR/MAR unit tests
│   └── test_classifier.py      # State machine and transition tests
└── md_files/                   # System documentation & memories
    ├── README.md               # Project documentation
    ├── PRD.md                  # Product Requirements Document
    ├── Architecture.md         # Detailed architectural specifications
    ├── Design.md               # Technical design & schema document
    ├── Rules.md                # Engineering standards & constraints
    └── memory.md               # Development memory & decision journal
```

---

## 🚀 Installation & Getting Started

### Prerequisites
- **Python:** 3.10 or higher
- **Hardware:** Standard USB or built-in webcam
- **OS:** Windows 10/11, macOS, or Linux

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/Deathsythe69/Drowsiness_Detection_system.git
cd Drowsiness_Detection_system

# Create virtual environment
python -m venv .venv

# Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

---

## ⚙️ Configuration Guide (`config.yaml`)

Edit parameters directly or adjust them via the in-app **Settings** menu:

```yaml
camera:
  index: 0                       # Webcam device index (0 for default)

thresholds:
  ear_threshold: 0.21            # Eye Aspect Ratio closure threshold
  ear_consec_frames: 20          # Consecutive frames before trigger (~1 sec @ 20 FPS)
  mar_threshold: 0.60            # Mouth Aspect Ratio yawn threshold
  mar_consec_frames: 15          # Consecutive frames for yawn confirmation
  fatigue_warning_score: 40.0    # Score threshold for visual warning
  fatigue_alert_score: 75.0      # Score threshold for auditory alarm

alerts:
  sound_file: "assets/alarm.wav" # Path to alarm audio file
  volume: 0.8                    # Audio output level (0.0 - 1.0)

ui:
  show_landmarks: true           # Draw facial mesh points over webcam feed
  theme: "dark"                  # Visual theme mode
```

---

## 📊 Session Analytics & Local Storage

All telemetry is recorded in a local, zero-config SQLite database (`drowsiness.db`).

### Schema Overview:
- **`sessions`**: `id`, `user_label`, `start_time`, `end_time`, `baseline_ear`
- **`events`**: `id`, `session_id`, `timestamp`, `event_type`, `ear_value`, `mar_value`

---

## 🔒 Privacy & Security

1. **100% Local Inference:** No video frames, images, or audio clips are stored or transmitted.
2. **Zero Cloud Dependencies:** Operates completely offline without external telemetry.
3. **Transparent Logging:** Only derived numerical values (EAR, MAR, timestamps) are persisted locally.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

Code formatting & linting standards:
```bash
black --line-length 100 .
ruff check .
```

---

## 🗺️ Roadmap

- [x] Real-time 468-point facial landmark detection (MediaPipe).
- [x] EAR/MAR computation with temporal smoothing.
- [x] Audio alarm with auto-synthesized wave asset fallback.
- [x] Cognitive math puzzle alarm dismissal.
- [x] SQLite local persistence and session summary charts.
- [ ] Head pose estimation (nodding / distraction detection via `solvePnP`).
- [ ] Focus / Pomodoro study timer integration.
- [ ] Exportable PDF session reports.
- [ ] Standalone executable builds (PyInstaller).

---

## 👥 License & Authors

- **Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))
- **Contributor:** Aditya Jena
- **License:** MIT License