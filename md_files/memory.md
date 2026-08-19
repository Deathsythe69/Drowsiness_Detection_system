# 🧠 Project Memory & Technical Journal
## Drowsiness & Attention Detection System

**Last Updated:** August 2026  
**Status:** Active Development (v1.0 MVP Implemented)  
**Primary Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))  

---

## 1. Project Context & Vision

The **Drowsiness & Attention Detection System** is an offline, privacy-first desktop application designed to detect user fatigue, micro-sleeps, and attention loss during extended computer sessions (remote working, online study, exam proctoring). It functions in real time using consumer-grade webcams and standard CPU hardware without sending data to the cloud.

---

## 2. Architectural Decision Records (ADRs)

| ID | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| **ADR-001** | **MediaPipe Face Mesh for landmark detection** | 468 3D landmarks, ultra-lightweight CPU execution (≥20 FPS), seamless pip install without C++ compilation requirements. | `dlib` 68-point predictor (heavyweight, C++ compilation hurdles on Windows). |
| **ADR-002** | **PyQt6 for Desktop GUI** | Rich widget ecosystem, smooth QSS styling, high-performance canvas updates, and thread-safe Qt Signal/Slot communication. | `Tkinter` (limited styling/performance), `Electron` (excessive resource footprint). |
| **ADR-003** | **Decoupled 3-Thread Architecture** | Decouples frame capture (`Thread 1`), landmark/classifier processing (`Thread 2`), and Qt UI rendering (`Main Thread`) via queues and signals to ensure 0 UI freezing. | Single-threaded OpenCV `imshow` loop (causes UI blocking and lag). |
| **ADR-004** | **Cognitive Math Puzzle for Alarm Dismissal** | Forces active cognitive engagement to dismiss auditory alerts, ensuring users cannot reflexively dismiss alarms while half-asleep. | Simple single-click "Dismiss" button. |
| **ADR-005** | **Zero-Dependency Alarm Wave Synthesis** | Uses Python standard library (`wave`, `struct`, `math`) to generate a default alarm tone if `assets/alarm.wav` is absent, preventing runtime crashes. | Mandatory external WAV asset distribution or third-party audio generator. |
| **ADR-006** | **Privacy-by-Design Local SQLite Storage** | Zero raw frames or image buffers are stored. Only derived numerical telemetry (EAR, MAR, timestamps, event flags) are written to local `drowsiness.db`. | Flat CSV files (harder to query for analytics), Cloud sync (violates privacy principle). |

---

## 3. Core Algorithms & Threshold Specifications

### 3.1 Eye Aspect Ratio (EAR)
Calculated from 6 landmark points per eye:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- **Default Baseline:** $\approx 0.25 - 0.35$ (calibrated dynamically during the first 10–15 seconds).
- **Closure Threshold:** `0.21` (configurable in `config.yaml`).
- **Trigger Duration:** $\ge 20$ consecutive frames ($\approx 1.0\text{s}$ at 20 FPS).

### 3.2 Mouth Aspect Ratio (MAR) — Yawn Detection
Calculated from 8 inner/outer mouth landmarks:
$$\text{MAR} = \frac{\|p_2 - p_8\| + \|p_3 - p_7\| + \|p_4 - p_6\|}{2 \cdot \|p_1 - p_5\|}$$
- **Yawn Threshold:** `0.60` (sustained for $\ge 15$ consecutive frames).
- **Role:** Increments rolling fatigue score rather than firing alarms directly.

### 3.3 State Classifier & Hysteresis
- **State Machine States:** `AWAKE` $\rightarrow$ `DROWSY_WARNING` $\rightarrow$ `DROWSY_ALERT`.
- **Hysteresis:** State transitions back to `AWAKE` only after continuous sustained open-eye frames, preventing alert flickering.

---

## 4. Codebase Map & Module Responsibilities

```
Drowsiness_Detection_system/
├── main.py                  # App entry point, config loader, default sound generator, QSS stylesheet
├── config.yaml              # Central YAML configuration
├── requirements.txt         # Runtime dependencies (opencv-python, mediapipe, PyQt6, PyYAML, matplotlib)
├── core/
│   ├── capture.py           # VideoCapture worker thread with frame queue
│   ├── landmarks.py         # MediaPipe FaceMesh wrapper returning normalized landmark arrays
│   ├── features.py          # Pure mathematical calculations for EAR, MAR, euclidean distances
│   ├── classifier.py        # FatigueState FSM, score accumulator, and transition logic
│   └── alerts.py            # QSoundEffect / audio playback and sine-wave audio generator
├── storage/
│   ├── db.py                # SQLite database manager, table initializations, and logging queries
│   └── models.py            # Dataclasses representing Session and Event records
├── ui/
│   ├── main_window.py       # Live monitor screen, real-time HUD, video canvas, and status cards
│   ├── math_puzzle_window.py# Interactive arithmetic modal required for alarm dismissal
│   ├── settings_window.py   # Parameter configuration dialog for thresholds, volume, camera index
│   └── summary_window.py    # Post-session dashboard with embedded matplotlib charts
├── tests/
│   ├── test_features.py     # Unit tests verifying mathematical correctness of EAR/MAR
│   └── test_classifier.py   # Unit tests verifying FSM state transitions and edge cases
└── md_files/
    ├── README.md            # User-facing README with image placeholders
    ├── PRD.md               # Product Requirements Document
    ├── Architecture.md      # System Architecture & data flow
    ├── Design.md            # Concrete module specifications & configuration schema
    ├── Rules.md             # Developer guidelines, code standards & testing policies
    └── memory.md            # (This file) Living project memory & decision log
```

---

## 5. Active Configuration Schema (`config.yaml`)

```yaml
camera:
  index: 0
thresholds:
  ear_threshold: 0.21
  ear_consec_frames: 20
  mar_threshold: 0.60
  mar_consec_frames: 15
  fatigue_warning_score: 40.0
  fatigue_alert_score: 75.0
alerts:
  sound_file: "assets/alarm.wav"
  volume: 0.8
ui:
  show_landmarks: true
  theme: "dark"
```

---

## 6. Engineering Conventions & Invariants

1. **Layer Separation:**
   - `core/` must never import from `ui/`.
   - `storage/` is the sole direct database interface.
2. **Thread Safety:**
   - No OpenCV or MediaPipe calls inside Qt GUI event handlers.
   - Worker-to-UI communication strictly via `pyqtSignal`.
3. **Privacy Invariant:**
   - Never persist raw frames or screenshots. Only derived metrics (EAR, MAR, timestamps) are stored.
4. **Testing Invariant:**
   - Any math or state transition modifications must have accompanying tests in `tests/`.

---

## 7. Current Implementation Milestones

- [x] **Milestone 1:** Core landmark detection & EAR/MAR feature extraction with MediaPipe.
- [x] **Milestone 2:** Finite State Machine (AWAKE, WARNING, ALERT) with temporal debounce.
- [x] **Milestone 3:** Thread-isolated PyQt6 UI with dark theme QSS styling and real-time landmark mesh HUD.
- [x] **Milestone 4:** Dynamic Baseline Calibration and Cognitive Math Puzzle alarm dismissal modal.
- [x] **Milestone 5:** SQLite local persistence and post-session matplotlib summary graphs.
- [x] **Milestone 6:** Comprehensive documentation suite in `md_files/` (`README.md`, `memory.md`, `PRD.md`, `Architecture.md`, `Design.md`, `Rules.md`).

---

## 8. Post-MVP & Future Considerations

- **Head Pose Estimation:** Integrate `cv2.solvePnP` to detect head nodding/pitch angle as a third fatigue factor.
- **Focus Timer / Pomodoro:** Embed customizable work/break interval timers into the main monitor window.
- **Session Report Export:** Add PDF export functionality for post-session reports using `reportlab`.
- **Standalone Installer:** Bundle via PyInstaller for single-click Windows `.exe` and macOS `.app` execution.
