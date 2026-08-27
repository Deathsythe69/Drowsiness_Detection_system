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

### 2.5 Eyewear Classification & Through-Reflection Enhancement (`core/eyewear_detector.py`)
- Automatically classifies driver ocular state into `NONE` (bare eyes), `REGULAR_GLASSES` (clear/prescription), and `SUNGLASSES` (dark/tinted lenses).
- **Ocular-to-Skin Luminance Ratio**: Computes $\frac{\bar{L}_{\text{eye\_region}}}{\bar{L}_{\text{cheek\_skin}}}$ combined with texture contrast to detect dark sunglasses.
- **Specular Glare & Frame Edge Detection**: Measures high-intensity specular highlights ($I > 218$) and nose bridge edge gradients characteristic of clear glass lenses.
- **15-Frame Hysteresis Deque**: Employs rolling window majority voting to eliminate frame-to-frame classification jitter.
- **Through-Reflection Inpainting & CLAHE**: Automatically suppresses specular lens reflections and restores eyelid/iris edge definition on clear glasses crops.

### 2.6 FSM Classifier, PERCLOS, Calibration & Safety Latching (`core/classifier.py`)
- **Startup Baseline Calibration**:
  - Automatically captures median EAR over the initial 90 frames (~3s at 30 FPS) while keeping the system in a safe AWAKE state.
  - Dynamically calculates the driver's personalized eye-closure threshold (`baseline_ear * 0.72`), eliminating false alerts caused by natural variation, illnesses, singing, or narrow resting eye shapes.
- **Direct Eye Tracking Mode (Bare Eyes & Clear Glasses)**:
  - **Time-Based Rolling PERCLOS**: Computes Percentage of Eye Closure ratio over time-based windows (30s short, 120s long; e.g. 900 and 3600 frames at 30 FPS) with a 50% window fill requirement to prevent startup false alarms.
  - **Damped Fatigue Accumulation**: Per-frame eye-closed fatigue increment is reduced to `+0.55` (configurable via `eye_closed_fatigue_increment`), ensuring normal blinks and brief glances do not snowball into false alerts.
  - **Glasses Glare EMA Guard**: When specular reflection or glare is detected on clear lenses (`is_glare_occluded = True`), the Exponential Moving Average (EMA) updates for EAR are frozen to preserve the last known valid baseline. Emits a `reduced_confidence` event and activates secondary MAR + Head Pose surrogate scoring during extended glare periods.
  - **Rolling Yawn Accumulator**: Maintains a rolling 5-minute (300s) queue of confirmed driver yawns.
  - Evaluates fatigue accumulation from prolonged eye closure, PERCLOS droop, yawn frequency ($\ge 3$ in 5m), and downward head drooping ($\text{Pitch} < -18^\circ$).
- **Sunglasses Surrogate Fatigue Mode (Dark Glasses)**:
  - Bypasses raw EAR and PERCLOS closure increments to prevent false alarms caused by dark tinted lenses.
  - **Surrogate Physiological Signals**: Tracks Mouth Aspect Ratio (MAR yawn kinetics & sustained slack jaw $+0.60$/frame) and rolling 5m yawn frequency ($\ge 2$ triggers Warning, $\ge 3$ triggers Alert).
  - **Surrogate Kinematic Signals**: Evaluates solvePnP downward head nodding ($\text{Pitch} < -20^\circ \implies +0.85$/frame up to alert level) and lateral head wobble/distraction ($|\text{Yaw}| > 30^\circ$).
  - **Compound Drowsiness Synergy**: Combined mouth opening ($MAR > 0.45$) + head drooping down immediately triggers `DROWSY_ALERT`.
- **Safety Alarm Latching**:
  - Once `DROWSY_ALERT` triggers, the FSM sets `alarm_latched = True`, locking the system into the alert state even if the driver briefly opens their eyes.
  - The buzzer continues sounding until an explicit reset occurs via `admin_reset()`, triggered by `MathPuzzleDialog` (cognitive proof of alertness), `AdminOverrideDialog` (supervisor PIN), or the remote admin `MUTE` API command.

### 2.6 Vectorized Neural Eye Openness Inference (`core/eye_classifier.py`)
- Pure NumPy vectorized Convolutional Neural Network employing `np.lib.stride_tricks.sliding_window_view` (`im2col`) and matrix multiplication (`np.dot` / BLAS GEMM).
- Replaces thousands of slow Python loops with <0.9ms CPU forward passes.
- Batched dual-eye method `predict_both_eyes(crop_l, crop_r)` processes both eyes simultaneously in a single matrix operation.

### 2.7 Continuous Active Learning & Live Auto-Trainer (`core/live_trainer.py`)
- Automatically captures hard edge-case eye crops (e.g. night-time closed/open eyes during driving alerts) into categorized dataset bins (`live_dataset/day_samples` and `live_dataset/night_samples`).
- Background asynchronous trainer (`LiveAutoTrainer`) trains lightweight CNN weights with synthetic data augmentation and immediately hot-reloads them into memory without restarting the application.

### 2.8 Blackbox Video Evidence Recorder (`core/evidence_recorder.py`)
- Continuous circular frame buffer retaining the past 3.0 seconds (pre-buffer) in RAM.
- When a `DROWSY_ALERT` occurs, records the pre-buffer plus 5.0 seconds of post-buffer video to MP4 format alongside JSON telemetry manifests for fleet incident analysis.

### 2.9 Non-Blocking Asynchronous SQLite Logger (`ui/main_window.py: AsyncDBLogger`)
- Uses a background worker thread and thread-safe `queue.Queue` to execute all SQLite database operations asynchronously.
- Eliminates GUI thread disk I/O stalls, ensuring consistent 60 FPS UI rendering and zero video stuttering.

### 2.10 Wi-Fi Network Discovery & QR Code Generator (`core/remote_admin.py`, `ui/qr_dialog.py`)
- Automatically discovers Wi-Fi and hotspot network interfaces using `psutil`.
- Generates dynamic, high-contrast QR code image streams via `qrcode`.
- Serves QR codes via the web endpoint `/qr` and in the interactive desktop modal `WiFiAccessDialog`.
- Enables mobile phones connected to the same Wi-Fi network or car hotspot to point their camera at the screen and open the admin control dashboard instantly.

### 2.11 Storage & Persistent Profiles (`storage/`)
- Manages `sessions`, `events`, and `user_profiles` in SQLite (`drowsiness_tracker.db`).
- Auto-migrates database schemas to support audit metadata.
- Allows saving and loading personalized calibration profiles across sessions.

### 2.12 Shared State Bridge (`core/shared_state.py`)
- Thread-safe singleton using `threading.Lock` for concurrent read/write access.
- Holds current system metrics (state, EAR, MAR, PERCLOS, fatigue score, head pose, vehicle motion state, buzzer status, session timer, shoulder angle, slouch status, stillness rigidity, and pose inference staleness).
- Queues buzzer commands (mute/unmute) from the remote admin and delivers them to the UI thread for execution.
- Maintains a 50-event rolling buffer for the remote dashboard event log.

### 2.13 Full-Body Posture Module (`core/posture.py`)
- Employs MediaPipe Pose (`model_complexity=0`, `enable_segmentation=False`) on upper-body landmarks (shoulders, ears, hips).
- Computes shoulder-line tilt angle $\theta_{\text{shoulder}}$ for lateral slouch/lean detection.
- Computes head-to-shoulder vertical droop delta combined with `solvePnP` 3D Pitch angle to identify torso collapse and slouching during fatigue.
- Computes rolling movement variance over an observation window to identify rigid microsleep stillness.
- Validates poses against the primary driver ROI to reject passenger bodies.
- Integrates into `StateClassifier` as a strictly secondary fatigue contributor and streams metrics to the live HUD and remote admin panel.
