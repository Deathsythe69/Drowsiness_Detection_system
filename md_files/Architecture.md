# System Architecture
## Drowsiness & Attention Detection System (v2.3)

---

## 1. High-Level Multi-Threaded Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                      Desktop App                                       │
│                                                                                        │
│  ┌────────────┐       ┌──────────────────────┐                                        │
│  │   Webcam   │──────▶│ Capture Thread       │                                        │
│  │ (Hardware) │       │ (Thread-Safe Queue)  │                                        │
│  └────────────┘       └──────────┬───────────┘                                        │
│                                  │                                                    │
│                  ┌───────────────┴───────────────┐                                    │
│                  ▼                               ▼                                    │
│       ┌────────────────────┐          ┌──────────────────────┐                        │
│       │ Low-Light / CLAHE  │          │ Vehicle Motion       │                        │
│       │ & Glare Processor  │          │ Detector (Background │                        │
│       └──────────┬─────────┘          │ Optical Flow Differ) │                        │
│                  │                    └──────────┬───────────┘                        │
│                  ▼                               │                                    │
│       ┌────────────────────┐                     │                                    │
│       │ MediaPipe Multi-   │                     │                                    │
│       │ Face Mesh (ROI)    │                     │                                    │
│       │ Driver vs Passenger│                     │                                    │
│       └──────────┬─────────┘                     │                                    │
│                  │                               │                                    │
│                  ▼                               │                                    │
│       ┌────────────────────┐                     │                                    │
│       │ Feature Extractor  │                     │                                    │
│       │ (Driver EAR, MAR,  │                     │                                    │
│       │  solvePnP Posture) │                     │                                    │
│       └──────────┬─────────┘                     │                                    │
│                  │                               │                                    │
│                  ▼                               │                                    │
│       ┌────────────────────┐                     │                                    │
│       │ FSM Classifier     │                     │                                    │
│       │ (Rolling Yawns,    │                     │                                    │
│       │  Driver Posture)   │                     │                                    │
│       └──────────┬─────────┘                     │                                    │
│                  │                               │                                    │
│                  └───────────────┬───────────────┘                                    │
│                                  │                                                    │
│       ┌──────────────────────────┼──────────────────────────┐                         │
│       ▼                          ▼                          ▼                         │
│ ┌───────────────┐        ┌──────────────┐            ┌──────────────────┐             │
│ │ Gated Alert   │        │ Storage      │            │ SharedState      │             │
│ │ System (Sound │        │ (SQLite: DB, │            │ (Thread-Safe     │             │
│ │ if Moving)    │        │  Profiles)   │            │  Metrics Bridge) │             │
│ └───────┬───────┘        └──────┬───────┘            └────────┬─────────┘             │
│         │                       │                             │                       │
│         └───────────────────────┤                             │                       │
│                                 │                             ▼                       │
│                                 │                ┌────────────────────────┐           │
│                                 │                │ Remote Admin Flask     │           │
│                                 │                │ Web Server (Daemon)    │           │
│                                 │                │ ── Wi-Fi / Hotspot ──  │           │
│                                 │                │ PIN Auth + Dashboard   │           │
│                                 │                │ Buzzer Mute/Unmute API │           │
│                                 │                │ Dynamic QR Code (/qr)  │           │
│                                 │                └────────────┬───────────┘           │
│                                 │                             │                       │
│                                 ▼                             ▼                       │
│                      ┌──────────────────────┐     ┌──────────────────────┐            │
│                      │ PyQt6 UI Layer (QSS) │     │ Mobile Phone Camera  │            │
│                      │ Live HUD | Telemetry │◀────│ (Instant QR Connect  │            │
│                      │ Motion | QR Dialog   │     │  over Wi-Fi/Hotspot) │            │
│                      └──────────────────────┘     └──────────────────────┘            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Low-Light & Glare Preprocessing (`core/preprocessing.py`)
- Analyzes frame luminance. When $L < 65$, applies CLAHE on the L-channel in LAB space and gamma brightening ($\gamma = 0.6$).
- Identifies saturated specularity spots on glasses lenses and dampens glare artifacts.

### 2.2 Vehicle Motion Detector (`core/motion_detector.py`)
- Extracts peripheral frame strips (top 15%, bottom 10%, left 15%, right 15%) corresponding to the windshield and windows.
- Calculates absolute frame-to-frame pixel difference across Gaussian-blurred grayscale frames.
- Uses a rolling decision window (e.g. 15 frames) to classify vehicle state as `DRIVING` vs `PARKED / STOPPED`.
- Acts as a hardware-free motion gate for auditory alarm dispatching.

### 2.3 Multi-Face Detection & Driver Isolation (`core/landmarks.py`)
- MediaPipe Face Mesh configured for up to 4 concurrent faces.
- Primary driver is selected based on bounding box size and central/driver Region of Interest (ROI).
- Passengers are labeled `PASSENGER (NO ALERTS)` and rendered with dimmed boxes without feeding into alerting logic.

### 2.4 Feature Extraction & Head Pose (`core/features.py`)
- Computes EAR across 6 eye landmarks and MAR across 8 inner/outer lip landmarks on the primary driver only.
- Solves Perspective-n-Point (`cv2.solvePnP`) using 6 canonical 3D facial anchors (nose tip, chin, eye corners, mouth corners) to extract Euler rotation angles (Pitch, Yaw, Roll).

### 2.5 FSM Classifier & Yawn Escalation (`core/classifier.py`)
- Maintains a rolling 5-minute (300s) queue of confirmed driver yawns.
- Evaluates fatigue accumulation from prolonged eye closure, yawn frequency ($\ge 3$ in 5m), and downward head drooping ($\text{Pitch} < -18^\circ$).
- Implements hysteresis buffers to prevent rapid alert oscillation.

### 2.6 Wi-Fi Network Discovery & QR Code Generator (`core/remote_admin.py`, `ui/qr_dialog.py`)
- Automatically discovers Wi-Fi and hotspot network interfaces using `psutil`.
- Generates dynamic, high-contrast QR code image streams via `qrcode`.
- Serves QR codes via the web endpoint `/qr` and in the interactive desktop modal `WiFiAccessDialog`.
- Enables mobile phones connected to the same Wi-Fi network or car hotspot to point their camera at the screen and open the admin control dashboard instantly.

### 2.7 Storage & Persistent Profiles (`storage/`)
- Manages `sessions`, `events`, and `user_profiles` in SQLite (`drowsiness_tracker.db`).
- Auto-migrates database schemas to support audit metadata.
- Allows saving and loading personalized calibration profiles across sessions.

### 2.8 Shared State Bridge (`core/shared_state.py`)
- Thread-safe singleton using `threading.Lock` for concurrent read/write access.
- Holds current system metrics (state, EAR, MAR, fatigue score, head pose, vehicle motion state, buzzer status, session timer).
- Queues buzzer commands (mute/unmute) from the remote admin and delivers them to the UI thread for execution.
- Maintains a 50-event rolling buffer for the remote dashboard event log.
