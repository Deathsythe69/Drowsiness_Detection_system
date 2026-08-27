# 👁️ Real-Time Drowsiness & Attention Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00C0A3.svg)](https://developers.google.com/mediapipe)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52.svg?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![Tests](https://img.shields.io/badge/Tests-100%2F100%20Passing-brightgreen.svg)](#testing--quality-assurance)

An intelligent, automotive-grade, privacy-first desktop application that monitors a user's facial state, eye closure kinetics, eyewear state (bare eyes, regular glasses, sunglasses), and posture in real time via standard webcam feeds. It continuously evaluates automotive-standard **PERCLOS**, dual-eye deep neural openness, rolling yawn frequency (MAR), head nodding/slouching (solvePnP 3D Head Pose), and cabin multi-occupant conditions to detect fatigue before safety or productivity is compromised.

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
- [Dataset Sourcing & Benchmarking](#-dataset-sourcing--benchmarking)
- [Privacy & Security](#-privacy--security)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License & Authors](#-license--authors)

---

## 🌟 Overview & Motivation

Extended screen time during study sessions, software development, online exams, and driving leads to fatigue-induced micro-sleeps and reduced cognitive vigilance.

This system provides a **100% offline, edge-computed solution** designed specifically for:
- 🎓 **Students & E-Learners:** Track study session alertness and avoid burnout.
- 💻 **Remote Professionals:** Monitor focus during extended work blocks with customizable break triggers.
- 🚗 **Vehicle & Workstation Inattention Monitoring:** Isolate the driver/primary operator from passengers and detect head nodding, slow eye closure, and acute fatigue.

---

## ✨ Key Features (v2.6)

- 🎯 **Startup Baseline Calibration:** Automatically captures driver's resting EAR/MAR over 3 seconds (90 frames) on startup to personalize eye closure thresholds (`baseline_ear * 0.72`), eliminating false alerts from natural blinks, singing, or narrow eyes.
- 🔒 **Safety Alarm Latching & Admin Reset:** Once a severe drowsiness alert triggers, the alarm latches ON indefinitely until authorized cognitive dismissal (Math Puzzle), supervisor PIN override, or remote admin MUTE command.
- 👓 **Intelligent Eyewear Detection, Glare Protection & Through-Reflection Vision:** Automatically detects **No Glasses**, **Regular Glasses**, or **Dark Sunglasses**. For regular glasses, freezes EMA EAR smoothing during reflection glare to prevent corrupted 0.0 values, flags `reduced_confidence`, and uses surrogate MAR/Pose metrics.
- 🕶️ **Adaptive Sunglasses Surrogate Fatigue Mode:** Dark sunglasses occlude eyes without triggering false closure alarms. Automatically switches to surrogate physiological tracking via **MAR (yawn duration & frequency)** and **solvePnP 3D Head Pose (Pitch nodding & Yaw drift)**.
- ⚡ **Real-Time Landmark Tracking & Vectorized Neural Inference:** Powered by MediaPipe 468-point Face Mesh combined with an ultra-fast vectorized Low-Light Eye CNN (`im2col` + BLAS GEMM) running in **<0.9ms** on standard CPUs ($\ge 30$ FPS).
- 👁️ **Automotive-Standard PERCLOS Metric:** Tracks Percentage of Eye Closure over time-based rolling deques (30s short, 120s long) with a 50% fill threshold to reliably catch slow eyelid droop and gradual fatigue buildup before microsleeps.
- 🌙 **Adaptive Low-Light Robustness:** Automatically detects low luminance ($L < 65$) and applies CLAHE + Gamma ($\gamma = 0.6$) shadow brightening and eyeglasses glare suppression.
- 👥 **Multi-Person Detection & Driver-Only Alerting:** Tracks multiple faces but isolates the primary driver inside the central Region of Interest; passengers are monitored visually without triggering false alarms.
- 🚗 **Vehicle Motion Detection & Driving Gating:** Uses background peripheral optical flow to detect vehicle motion. Audible buzzer alarms sound **only when the driver is drowsy while driving**, and are automatically muted when parked or stationary.
- 🧠 **Continuous Active Learning & Hot-Reload:** Automatically harvests and tags edge-case driver eye crops during live usage and enables 1-click model self-training with immediate in-memory weight hot-reloading.

- 📹 **Automated Blackbox Evidence Recorder:** Circular pre/post rolling frame buffer automatically records and persists high-resolution MP4 video clips whenever a critical drowsiness alert is triggered.
- ⚡ **Zero-Latency Async SQLite Worker:** Dedicated background daemon queue decouples all database disk I/O from the Qt GUI thread to guarantee 0ms frame lag.
- 📱 **Wi-Fi & Hotspot Mobile QR Code Access:** Embedded Flask web server accessible across Wi-Fi, Ethernet, and Hotspots with dynamic high-resolution QR codes (`WiFiAccessDialog`).
- 🥱 **Rolling-Window Yawn Frequency Escalation:** Tracks confirmed yawns over a rolling 5-minute (300s) window. If $\ge 3$ yawns occur within 5 minutes, state escalates to warning/alert.
- 📐 **3D Head Pose & Posture Estimation:** Calculates Pitch, Yaw, and Roll using `cv2.solvePnP` against a 3D canonical model, penalizing head drooping ($\text{Pitch} < -18^\circ$) and inattention.
- 👤 **Persistent User Profiles:** Stores personalized calibration presets (e.g. `"User_Normal"`, `"User_Glasses"`) in SQLite.
- 🔋 **Battery Health & Camera Resilience:** Monitors laptop battery state with warnings at $<20\%$; gracefully auto-saves sessions if the camera disconnects.
- 🔐 **Dual-Tier Alarm Dismissal:** Cognitive math verification puzzle for users, plus supervisor Admin PIN override (`"1234"`).
- 🌐 **Remote Admin Web Panel:** Supervisors can monitor live telemetry, mute/unmute buzzer, and view event logs from any LAN/Wi-Fi connected device.

---

## 📸 Visual Walkthrough & Screenshots

> **Note:** Upload your application screenshots into the `assets/screenshots/` folder to populate these views.

### 1. Live Monitoring Interface
*Webcam stream with real-time landmark mesh, EAR/MAR gauges, session timer, battery indicator, and dynamic state card.*

<!-- IMAGE PLACEHOLDER: LIVE MONITOR -->
<p align="center">
  <img src="../assets/screenshots/live_monitor.png" alt="Live Monitor Interface" width="85%" />
  <br />
  <em>Figure 1: Main live monitor screen with real-time facial mesh, head pose, and telemetry HUD.</em>
</p>

---

### 2. Multi-Face Cabin Driver vs. Passenger Isolation
*Isolates driver within central ROI; non-primary passengers are tagged with gray bounding boxes and excluded from alert triggers.*

<!-- IMAGE PLACEHOLDER: MULTI-FACE -->
<p align="center">
  <img src="../assets/screenshots/multi_face_isolation.png" alt="Multi-Face Driver Isolation" width="85%" />
  <br />
  <em>Figure 2: Multi-person detection showing active driver bounding box and ignored passenger bounds.</em>
</p>

---

### 3. Cognitive Math Puzzle Wake-Up Modal
*Ensures the user is genuinely alert before silencing alarms and returning to active monitoring.*

<!-- IMAGE PLACEHOLDER: MATH PUZZLE -->
<p align="center">
  <img src="../assets/screenshots/math_puzzle.png" alt="Cognitive Math Puzzle Modal" width="70%" />
  <br />
  <em>Figure 3: Interactive arithmetic puzzle required for user alarm dismissal.</em>
</p>

---

### 5. Admin Supervisor PIN Alarm Override
*Supervisor-authenticated modal to silence the buzzer without ending the session, with persistent audit logging.*

<!-- IMAGE PLACEHOLDER: ADMIN OVERRIDE -->
<p align="center">
  <img src="../assets/screenshots/admin_override.png" alt="Admin Override Modal" width="70%" />
  <br />
  <em>Figure 5: Secure PIN dialog for administrative alarm silencing.</em>
</p>

---

### 6. Remote Admin Web Dashboard (LAN)
*Browser-based admin panel accessible from any device on the same LAN. Supervisors can monitor live metrics and mute/unmute the buzzer remotely.*

<!-- IMAGE PLACEHOLDER: REMOTE ADMIN DASHBOARD -->
<p align="center">
  <img src="../assets/screenshots/remote_admin_dashboard.png" alt="Remote Admin Web Dashboard" width="85%" />
  <br />
  <em>Figure 6: Web-based remote admin dashboard with real-time metrics and buzzer control.</em>
</p>

---

### 7. Settings & Sensitivity Configuration
*Customizable EAR/MAR thresholds, low-light parameters, posture angles, camera selection, remote admin toggle, and Admin PIN.*

<!-- IMAGE PLACEHOLDER: SETTINGS PANEL -->
<p align="center">
  <img src="../assets/screenshots/settings_panel.png" alt="Settings and Configuration Panel" width="80%" />
  <br />
  <em>Figure 7: Configuration window with threshold sliders, remote admin settings, and hardware management.</em>
</p>

---

### 8. Post-Session Analytics Dashboard
*Comprehensive historical charts showing fatigue spikes, attention curves, and session summary metrics.*

<!-- IMAGE PLACEHOLDER: SESSION SUMMARY -->
<p align="center">
  <img src="../assets/screenshots/session_summary.png" alt="Session Summary and Analytics" width="85%" />
  <br />
  <em>Figure 8: Post-session analytics dashboard displaying EAR distribution over time.</em>
</p>

---

## 🏗️ Architecture & Detection Pipeline

```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Webcam Stream  │──────▶│ Capture Thread (Queue) │──────▶│ Low-Light / CLAHE    │
└─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                      │
                                                                      ▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  PyQt6 UI Loop  │◀──────│  FSM Classifier &      │◀──────│ Multi-Face Mesh &    │
│  (Signals/Slots)│       │  Rolling Yawn Tracker  │       │ solvePnP Head Pose   │
└────────┬────────┘       └───────────┬────────────┘       └──────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐       ┌────────────────────────┐
│ Audio Alarms &  │       │ SQLite Storage         │
│ Admin Overrides │       │ (Sessions & Profiles)  │
└─────────────────┘       └────────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Eye Aspect Ratio (EAR)
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

### 2. Mouth Aspect Ratio (MAR) & Yawn Frequency
$$\text{MAR} = \frac{\|p_2 - p_8\| + \|p_3 - p_7\| + \|p_4 - p_6\|}{2 \cdot \|p_1 - p_5\|}$$
$$\text{Rolling Frequency Check:} \quad N_{\text{yawns}}(\text{last 300s}) \ge 3 \implies \text{Escalate Alert}$$

### 3. Head Pose Estimation (Pitch, Yaw, Roll)
Maps 6 key 3D landmarks (Nose tip, Chin, Eye corners, Mouth corners) via `cv2.solvePnP` against a canonical 3D facial model to extract Euler rotation angles.

---

## 📂 Project Directory Structure

```
Drowsiness_Detection_system/
├── main.py                     # Application entry point & Qt launcher
├── config.yaml                 # Tunable thresholds, camera, admin, and alert settings
├── requirements.txt            # Python dependencies (OpenCV, MediaPipe, PyQt6, psutil, flask, etc.)
├── run.bat / run.ps1           # Quick launch scripts for Windows desktop
├── assets/                     # Sound files, icons, and screenshot assets
│   ├── alarm.wav               # Auto-synthesized or custom alert sound
│   └── screenshots/            # Place application screenshot images here
├── core/                       # Core CV and detection logic
│   ├── capture.py              # Threaded webcam frame grabber
│   ├── preprocessing.py        # Low-light CLAHE enhancement & glare reduction
│   ├── landmarks.py            # MediaPipe multi-face & driver ROI isolation
│   ├── features.py             # EAR, MAR, solvePnP Head Pose algorithms
│   ├── classifier.py           # Multi-signal FSM & rolling yawn frequency tracker
│   ├── motion_detector.py      # Background optical flow vehicle motion detector
│   ├── alerts.py               # Sound synthesizer & alarm dispatcher
│   ├── system_health.py        # Battery and hardware monitoring
│   ├── shared_state.py         # Thread-safe singleton metrics bridge
│   └── remote_admin.py         # Embedded Flask admin web server (LAN)
├── storage/                    # Local database layer
│   ├── db.py                   # SQLite connection, migrations, and queries
│   └── models.py               # Session, Event, and UserProfile data models
├── ui/                         # PyQt6 modern dark-themed GUI
│   ├── main_window.py          # Primary monitoring screen, HUD, motion, & remote admin integration
│   ├── admin_dialog.py         # Supervisor PIN alarm override modal (local)
│   ├── profile_dialog.py       # Persistent calibration profile manager
│   ├── math_puzzle_window.py   # Cognitive alarm verification puzzle
│   ├── settings_window.py      # Thresholds, motion sensitivity, remote admin toggle
│   ├── summary_window.py       # Session analytics & matplotlib graphs
│   └── qr_dialog.py            # Wi-Fi & Mobile QR Code Access Dialog
├── tests/                      # Automated test suite (54/54 passing)
│   ├── test_features.py        # EAR/MAR unit tests
│   ├── test_classifier.py      # State machine and transition tests
│   ├── test_preprocessing.py   # Brightness, CLAHE, and glare tests
│   ├── test_head_pose.py       # solvePnP pitch/yaw posture tests
│   ├── test_motion_detector.py # Vehicle motion detector tests
│   ├── test_yawn_frequency.py  # Rolling 5m yawn frequency escalation tests
│   ├── test_profiles.py        # UserProfile CRUD tests
│   ├── test_multi_face.py      # Driver ROI isolation tests
│   ├── test_shared_state.py    # SharedState singleton & thread-safety tests
│   ├── test_remote_admin.py    # Flask admin auth, API, QR code, & buzzer control tests
│   └── test_math_puzzle.py     # Arithmetic puzzle tests
└── md_files/                   # System documentation & memories
    ├── README.md               # Project documentation
    ├── PRD.md                  # Product Requirements Document (v2.3)
    ├── Architecture.md         # Detailed architectural specifications
    ├── Design.md               # Technical design, REST API, & schema document
    ├── Dataset_and_Benchmarking.md # Dataset sourcing & benchmarking plan
    ├── Rules.md                # Engineering standards & constraints
    └── memory.md               # Development memory & decision journal
```

---

## 🚀 Installation & Getting Started

### 1. Clone Repository & Setup Environment
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
# Or on Windows, double-click run.bat
```

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

All 87 unit tests validate:
- Eyewear classification (bare eyes, regular glasses with glare, dark sunglasses) and through-reflection enhancement
- Sunglasses surrogate physiological fatigue mode (bypassing EAR/PERCLOS false positives, escalating on MAR & solvePnP Head Pose)
- Automotive PERCLOS temporal accumulation and multi-signal alert escalation
- Vectorized Low-Light Eye CNN GEMM inference (<0.9ms) and dual-eye prediction
- Continuous active learning sample collector and background self-training hot-reload
- Circular pre/post buffer blackbox video evidence recording
- Non-blocking asynchronous SQLite queue logging (`AsyncDBLogger`)
- EAR / MAR calculation accuracy and eye closure debouncing
- Low-light CLAHE and glare suppression
- solvePnP head pose Pitch/Yaw angle derivation
- Rolling yawn frequency escalation
- Vehicle motion detection, peripheral optical differencing, and test driving simulation
- Multi-face driver ROI isolation & passenger ignoring
- Persistent UserProfile SQLite CRUD
- SharedState singleton, thread-safety, and buzzer command queue
- Remote Admin Flask panel auth, status API, buzzer control, QR code endpoint, and event log

---

## 👥 License & Authors

- **Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))
- **Contributor:** Aditya Jena
- **License:** MIT License