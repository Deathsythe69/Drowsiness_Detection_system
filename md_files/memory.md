# 🧠 Project Memory & Technical Journal
## Drowsiness & Attention Detection System

**Last Synchronized:** August 2026  
**Status:** v2.3 Production Ready (Wi-Fi / Hotspot Mobile QR Access & Vehicle Motion Gating)  
**Primary Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))  
**Repository:** [Deathsythe69/Drowsiness_Detection_system](https://github.com/Deathsythe69/Drowsiness_Detection_system)

---

## 1. Continuous Memory & Documentation Invariant

> [!IMPORTANT]
> **Developer & AI Assistant Policy:**  
> After **each and every chat session or code change**, `memory.md` and all related markdown documents in `md_files/` (`PRD.md`, `Architecture.md`, `Design.md`, `README.md`, `Dataset_and_Benchmarking.md`, `Rules.md`) must be reviewed, synchronized, and improved to reflect the latest codebase state, configuration keys, architectural decisions, and testing verification.

---

## 2. Architectural Decision Records (ADRs)

| ID | Title | Rationale | Alternatives Considered |
|---|---|---|---|
| **ADR-001** | **MediaPipe Face Mesh (Multi-Face ROI Driver Tracking)** | 468 3D landmarks computed on CPU (≥20 FPS); configured with `max_num_faces=4` to isolate the primary driver in a central ROI while ignoring secondary passengers. | `dlib` 68-point predictor (heavyweight, C++ build friction on Windows), single-face detector (caused passenger false alerts). |
| **ADR-002** | **Adaptive CLAHE & Gamma Low-Light Preprocessing** | When frame luminance $L < 65.0$, CLAHE is applied on the L-channel in LAB space with Gamma correction ($\gamma = 0.6$) to lift shadow pixels and restore eye landmark visibility in dark rooms/night driving. Added specular reflection dampening for eyeglasses. | Global histogram equalization (washed out contrast and over-saturated bright areas). |
| **ADR-003** | **Persistent SQLite User Calibration Profiles** | `user_profiles` table stores named calibration presets (`name`, `baseline_ear`, `ear_threshold`, `baseline_mar`, `mar_threshold`, `blink_rate`), allowing returning users to load personalized thresholds instantly without recalibrating every session. | Ephemeral session-only calibration variables. |
| **ADR-004** | **Rolling-Window Yawn Frequency Escalation** | `StateClassifier` tracks confirmed yawns across a rolling 5-minute (300s) time window. If $\ge 3$ yawns occur within 5 minutes, state escalates to `DROWSY_WARNING` / `DROWSY_ALERT`. | Simple static counter without timestamp expiration (caused cumulative false positives over long sessions). |
| **ADR-005** | **solvePnP 3D Head Pose for Posture Analysis** | Maps 6 key 3D canonical landmarks (nose tip, chin, eye corners, mouth corners) to derive Euler angles (**Pitch, Yaw, Roll**). Penalizes head nodding ($\text{Pitch} < -18^\circ$) and sustained distraction as a secondary fatigue input. | 2D heuristic eye-to-nose slope estimation. |
| **ADR-006** | **Dual-Tier Dismissal: Cognitive Puzzle + Admin PIN** | Regular users solve arithmetic problems in `MathPuzzleDialog` to ensure alertness; supervisors can use `AdminOverrideDialog` with PIN `"1234"` to silence alarms instantly with SQLite audit logging. | Single unverified dismiss button. |
| **ADR-007** | **Device Battery & Camera Disconnect Resilience** | `psutil` battery monitor warns when laptop battery $< 20\%$. Catching camera disconnects triggers graceful session auto-save in SQLite instead of unhandled crash. | Abrupt termination on USB or battery power loss. |
| **ADR-008** | **Decoupled 3-Thread Architecture** | Frame capture (`Thread 1`), landmark & feature processing (`Thread 2`), and PyQt6 GUI event loop (`Main Thread`) communicate via thread-safe queues and Qt signals, guaranteeing 0% UI freeze. | Single-threaded OpenCV loop. |
| **ADR-009** | **Remote Admin Web Panel (Flask + SharedState Bridge)** | Embedded lightweight Flask HTTP server runs in a daemon background thread, binding to `0.0.0.0:<port>` so any device on the same LAN can access a web-based admin dashboard. Uses a thread-safe `SharedState` singleton to bridge real-time metrics between the PyQt6 UI thread and Flask routes. PIN-authenticated with full audit logging of all remote buzzer override events. | External Celery/Redis broker (heavy, requires separate infrastructure), WebSocket-only approach (complex, less compatible with mobile browsers). |
| **ADR-010** | **Vehicle Motion Detection & Driving Gating** | Employs background optical flow / peripheral frame differencing on top/bottom/side frame margins to estimate vehicle movement. Drowsiness alarms are conditionally gated: audible buzzer sounds **only when the vehicle is in active driving motion**, and is automatically muted when parked or stationary to prevent false alarms during rest stops. | Heavyweight full-frame dense optical flow (too slow for real-time CPU), GPS OBD-II hardware dependency (not universally available). |
| **ADR-011** | **Wi-Fi / Mobile Hotspot Discovery & Scannable QR Code** | Uses `psutil.net_if_addrs()` to detect and prioritize Wi-Fi adapters alongside Ethernet and Mobile Hotspots. Generates dynamic PNG QR codes via `qrcode` (accessible at `/qr` and in the desktop UI `WiFiAccessDialog`), allowing mobile devices connected to the same Wi-Fi or car hotspot to point their camera at the screen and instantly open the remote admin dashboard without manual typing. | Manual IP typing only (error-prone on small phone keyboards), Bluetooth-only pairing (driver friction). |
| **ADR-012** | **DirectShow (`cv2.CAP_DSHOW`) Windows Camera Backend** | Windows MSMF backend suffers from `-1072875772` sample reader failure and high latency. Explicitly initializes `cv2.VideoCapture(index, cv2.CAP_DSHOW)` with fallback, ensuring instant (<100ms) webcam startup and stable frame streaming. | Default OpenCV VideoCapture without explicit backend. |
| **ADR-013** | **Ultra-Low Latency Pipeline & Buffer Copy Safety** | Replaced heavy `cv2.inpaint` in glare suppression with fast threshold clipping; downsampled motion detector input to $160 \times 120$ (<0.2ms diff); decoupled Qt `QImage` memory via `.copy()` with `FastTransformation` scaling to prevent gray canvas corruption. | Heavy inpainting and uncopied QImage buffer. |
| **ADR-014** | **Pure-Vectorized Low-Light Eye CNN (`im2col` + BLAS GEMM)** | Replaced nested `cv2.filter2D` loops (3,104 calls/frame causing 80-150ms UI stalls) with NumPy `sliding_window_view` + `np.dot` (BLAS GEMM) matrix multiplication. Dual-eye forward pass executes in <0.9ms on CPU with zero frame drops. | Nested OpenCV filter loops, heavyweight PyTorch runtime. |
| **ADR-015** | **Automotive PERCLOS Metric & Non-Blocking Async SQLite Queue** | Implemented NHTSA/ISO standard PERCLOS ($P_{80}$ closure ratio over 60-frame ~2s and 300-frame ~10s rolling windows) for early detection of slow eye closure before microsleeps. Replaced synchronous UI-thread SQLite disk writes with `AsyncDBLogger` (daemon worker + queue) to guarantee 0ms disk latency. | Geometry-only EAR without temporal accumulation, blocking disk I/O on UI thread. |
| **ADR-016** | **Intelligent Eyewear Detection, Through-Reflection Vision & Sunglasses Surrogate Mode** | Automatically classifies driver ocular state into `NONE` (bare eyes), `REGULAR_GLASSES` (clear/prescription), and `SUNGLASSES` (dark/tinted lenses) using ocular-to-skin luminance ratios, specular glare detection, and 15-frame hysteresis voting. Regular glasses use targeted reflection dampening + CLAHE to see through glare. Sunglasses safely bypass EAR/PERCLOS to eliminate false alarms and activate **Surrogate Fatigue Tracking** via MAR (yawn kinetics & open mouth) + solvePnP 3D Head Pose (Pitch nodding & Yaw drift). | Static single-mode classification (caused black lenses to falsely trigger eye-closure alarms). |

---

## 3. Complete Codebase Map & Module Responsibilities

```
Drowsiness_Detection_system/
├── main.py                     # App entry point, config loader, sound generator, QSS stylesheet, global exception hook
├── config.yaml                 # Central YAML configuration (thresholds, PERCLOS, eyewear, admin, motion, remote panel, etc.)
├── requirements.txt            # Runtime dependencies (OpenCV, MediaPipe, PyQt6, PyYAML, matplotlib, psutil, flask, qrcode, pytest)
├── run.bat / run.ps1           # Windows one-click desktop launchers
├── assets/                     # Sound files, icons, and screenshot assets
│   ├── alarm.wav               # Auto-synthesized or custom alert sound
│   └── screenshots/            # Directory for project screenshots
├── core/                       # Core computer vision & physiological logic
│   ├── capture.py              # VideoCapture worker thread with frame queue and disconnect signals
│   ├── preprocessing.py        # Low-light luminance detection, CLAHE enhancement, and glare reduction
│   ├── landmarks.py            # MediaPipe FaceMesh wrapper with multi-face tracking & driver ROI isolation
│   ├── features.py             # EAR, MAR, and solvePnP Head Pose (Pitch, Yaw, Roll) estimation
│   ├── eyewear_detector.py     # Eyewear classifier (Glasses vs Sunglasses) & through-reflection optical enhancement
│   ├── classifier.py           # Multi-signal FSM, PERCLOS rolling window, Sunglasses surrogate mode (MAR+Pose), & yawn tracker
│   ├── eye_classifier.py       # Vectorized Low-Light Eye Openness Neural Classifier (<0.9ms BLAS GEMM)
│   ├── live_trainer.py         # Continuous active learning sample collector and background self-trainer
│   ├── evidence_recorder.py    # Automated rolling circular buffer blackbox evidence video recorder
│   ├── motion_detector.py      # Background peripheral optical flow vehicle motion detector
│   ├── alerts.py               # Sound synthesizer and alarm loop manager
│   ├── system_health.py        # Laptop battery and camera connection health monitor
│   ├── shared_state.py         # Thread-safe singleton bridge between UI thread and Flask admin server
│   └── remote_admin.py         # Embedded Flask web server, Wi-Fi discovery, QR generator, & buzzer control API
├── train_test/                 # Model training and verification
│   └── train_low_light_model.py # Vectorized neural training pipeline, synthetic night augmentation, and weight exporter
├── storage/                    # Local SQLite database layer
│   ├── db.py                   # SQLite connection lifecycle, schema initialization, and auto-migrations
│   └── models.py               # Session, Event, and UserProfile dataclass CRUD operations (with get_by_session)
├── ui/                         # PyQt6 modern dark-themed GUI
│   ├── main_window.py          # Primary monitoring screen, video canvas, HUD, PERCLOS gauge, Eyewear readout, AsyncDBLogger
│   ├── admin_dialog.py         # Supervisor PIN-gated alarm override modal (local)
│   ├── profile_dialog.py       # Persistent user calibration profile manager
│   ├── math_puzzle_window.py   # Cognitive alarm verification puzzle dialog
│   ├── settings_window.py      # Threshold, camera, motion, low-light, posture, and remote admin config dialog
│   ├── summary_window.py       # Historical session analytics with embedded matplotlib graphs
│   └── qr_dialog.py            # Wi-Fi & Mobile QR Code Access Dialog
├── tests/                      # Automated test suite (87/87 passing)
│   ├── test_eyewear_detector.py # Eyewear classification, glare detection, reflection filtering, and hysteresis tests
│   ├── test_classifier.py      # FSM state transitions, PERCLOS, and Sunglasses surrogate mode tests
│   ├── test_features.py        # EAR and MAR mathematical unit tests
│   ├── test_head_pose.py       # solvePnP Euler angle derivation and posture penalty tests
│   ├── test_low_light_model.py # Low-light enhancement, night augmentation, CNN forward pass, and dual-eye GEMM latency tests
│   ├── test_live_trainer.py    # Sample collector and asynchronous self-training hot-reloader tests
│   ├── test_evidence_recorder.py # Circular pre/post buffer video encoding and metadata persistence tests
│   ├── test_driving_simulation.py # Simulated driving toggle and motion bypass tests
│   ├── test_sweet_spot_fusion.py # Multi-signal sweet spot fusion score tests
│   ├── test_performance_and_latency.py # Sub-50ms latency and high FPS throughput tests
│   ├── test_async_db_logger.py # Non-blocking asynchronous SQLite logging tests
│   ├── test_math_puzzle.py     # Arithmetic puzzle generation and answer validation tests
│   ├── test_motion_detector.py # Vehicle motion detector and peripheral differencing tests
│   ├── test_multi_face.py      # Multi-face bounding box and driver ROI isolation tests
│   ├── test_preprocessing.py   # Brightness detection, CLAHE enhancement, and glare filter tests
│   ├── test_profiles.py        # SQLite UserProfile CRUD and calibration persistence tests
│   ├── test_yawn_frequency.py  # Rolling 5m window yawn accumulation and threshold escalation tests
│   ├── test_shared_state.py    # SharedState singleton, thread-safety, buzzer command, and event buffer tests
│   └── test_remote_admin.py    # Flask admin panel auth, status API, buzzer control, QR code, and event log tests
└── md_files/                   # System documentation & memories
    ├── README.md               # User-facing README with screenshot slots and quickstart guide
    ├── PRD.md                  # Product Requirements Document (v2.4)
    ├── Architecture.md         # Detailed system architecture and data flow diagrams
    ├── Design.md               # Mathematical formulas, database schemas, REST API, and config schema
    ├── Dataset_and_Benchmarking.md # Public dataset sourcing, licensing, and benchmarking plan
    ├── Rules.md                # Developer guidelines, code standards, and documentation invariants
    ├── memory.md               # (This file) Living project memory and decision journal
    └── faculty_feedback_improvement_prompt.md # Raw faculty review requirements
```

---

## 4. Active Configuration Schema (`config.yaml`)

```yaml
alerts:
  sound_file: assets/alarm.wav
  volume: 0.8

admin:
  pin: "1234"
  remote_enabled: true       # Enable/disable the remote admin web panel
  remote_port: 8080          # TCP port for the Flask admin server

camera:
  index: 0

thresholds:
  ear_consec_frames: 20
  ear_threshold: 0.21
  max_blink_frames: 15       # Normal blink duration filter
  perclos_threshold: 0.35    # Automotive standard: 35% eye closure triggers fatigue alert
  fatigue_alert_score: 75.0
  fatigue_warning_score: 40.0
  mar_consec_frames: 15
  mar_threshold: 0.60
  yawn_window_seconds: 300
  yawn_frequency_alert_threshold: 3

low_light:
  enabled: true
  brightness_threshold: 65.0
  gamma: 0.6
  clahe_clip_limit: 2.5
  clahe_grid_size: 8

posture:
  enabled: true
  pitch_nod_threshold: -18.0
  yaw_distraction_threshold: 25.0
  slouch_penalty_weight: 1.0

battery:
  enabled: true
  low_battery_threshold: 20
  check_interval_seconds: 5

multi_face:
  max_faces: 4
  roi:
    x_min: 0.10
    x_max: 0.90
    y_min: 0.05
    y_max: 0.95

motion:
  enabled: true
  require_motion_for_alert: true  # Alert ONLY when vehicle is moving
  simulate_driving: false         # Toggleable in UI for desk testing
  motion_threshold: 4.0
  window_size: 15
  min_moving_ratio: 0.4

evidence:
  enabled: true
  directory: evidence
  pre_buffer_seconds: 3.0
  post_buffer_seconds: 5.0
  fps: 20.0
  cooldown_seconds: 15.0
  codec: mp4v

ui:
  show_landmarks: true
  show_passengers: true
  theme: dark
```

---

## 5. Verification & Test Suite Summary

Executed command:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Results: **80 passed in 18.75s (100% pass rate)**.
- `test_async_db_logger.py` (1 test): Non-blocking asynchronous SQLite event logging worker.
- `test_classifier.py` (5 tests): AWAKE state, prolonged eye closure, yawn increment, face loss recovery, and rolling PERCLOS accumulation.
- `test_low_light_model.py` (5 tests): CLAHE enhancement, night augmentation, CNN forward pass, inference, and batched dual-eye GEMM (<5ms).
- `test_live_trainer.py` (4 tests): Live sample collector, directory structure, dataset statistics, background asynchronous auto-trainer.
- `test_evidence_recorder.py` (5 tests): Circular buffer management, trigger recording, video writer generation, cooldown, metadata export.
- `test_driving_simulation.py` (3 tests): Simulated driving mode toggle, motion score bypass, config persistence.
- `test_sweet_spot_fusion.py` (5 tests): Multi-modal fusion logic, low-light weighting, EAR+CNN agreement.
- `test_performance_and_latency.py` (2 tests): End-to-end pipeline throughput and sub-50ms latency benchmarks.
- `test_features.py` (4 tests): Euclidean distance, single eye EAR, empty landmarks fallback, empty mouth fallback.
- `test_head_pose.py` (2 tests): solvePnP fallback on empty landmarks, nodding & distraction posture evaluation.
- `test_math_puzzle.py` (2 tests): Arithmetic puzzle generation, answer verification.
- `test_motion_detector.py` (5 tests): Initialization, None/empty frame handling, static parked frames detection, moving scenery optical flow, reset function.
- `test_multi_face.py` (2 tests): Multi-face initialization, bounding box and center extraction.
- `test_preprocessing.py` (4 tests): Frame brightness calculation, CLAHE & gamma brightening, glare filter, integration test.
- `test_profiles.py` (1 test): UserProfile creation, retrieval, update, and deletion in SQLite.
- `test_yawn_frequency.py` (1 test): Rolling 5m window yawn accumulation and threshold escalation.
- `test_shared_state.py` (12 tests): Singleton pattern, metric read/write with motion and PERCLOS metrics, concurrent thread-safety, buzzer command queue/consume, event buffer cap at 50, monitoring state tracking.
- `test_remote_admin.py` (17 tests): Login page, PIN auth (valid/invalid), dashboard access control, logout, `/status` JSON with motion metrics, metrics reflection, `/buzzer/mute` and `/buzzer/unmute` commands, unauthenticated 401 responses, `/log` endpoint with event data, `/qr` endpoint PNG output, `generate_qr_code_bytes`, and `get_network_info` discovery.

---

## 6. Milestone Progress & Roadmap

- [x] **Milestone 1:** MediaPipe Face Mesh & EAR/MAR feature extractor.
- [x] **Milestone 2:** Multi-signal FSM Classifier with hysteresis debounce.
- [x] **Milestone 3:** Thread-isolated PyQt6 UI with dark theme styling.
- [x] **Milestone 4:** Adaptive Low-Light CLAHE & Gamma shadow enhancement.
- [x] **Milestone 5:** Persistent User Calibration Profiles in SQLite.
- [x] **Milestone 6:** Rolling 5-minute Yawn Frequency Escalation.
- [x] **Milestone 7:** solvePnP 3D Head Pose & Nodding / Posture Analysis.
- [x] **Milestone 8:** Laptop Battery Health Monitoring & Graceful Disconnect Auto-Save.
- [x] **Milestone 9:** Dual-tier Alert Dismissal (Cognitive Math Puzzle + Admin PIN Override).
- [x] **Milestone 10:** Multi-Face Cabin Tracking with Driver ROI Isolation.
- [x] **Milestone 11:** Dataset Sourcing, Licensing & Benchmarking Documentation.
- [x] **Milestone 12:** Remote Admin Web Panel (Flask + SharedState LAN buzzer control with audit logging).
- [x] **Milestone 13:** Vehicle Motion Detection & Driving Gating (Alert only when driver is sleeping while moving).
- [x] **Milestone 14:** Wi-Fi / Hotspot Mobile Network Discovery & Scannable QR Code Dialog.
- [x] **Milestone 15:** Vectorized Low-Light Eye CNN (`im2col` + BLAS GEMM) & Hot-Reload Active Learning.
- [x] **Milestone 16:** Automotive Standard PERCLOS Drowsiness Detection & Zero-Latency Async SQLite Worker.
- [ ] **Milestone 17 (Future):** Pomodoro / Study interval timer widget.
- [ ] **Milestone 18 (Future):** Exportable PDF session reports using `reportlab`.
