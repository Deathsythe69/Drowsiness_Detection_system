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

---

## 3. Complete Codebase Map & Module Responsibilities

```
Drowsiness_Detection_system/
├── main.py                     # App entry point, config loader, default sound generator, QSS stylesheet, global exception hook
├── config.yaml                 # Central YAML configuration (thresholds, admin, motion, remote panel, etc.)
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
│   ├── classifier.py           # Multi-signal FSM, rolling yawn frequency tracker, and posture weighting
│   ├── motion_detector.py      # Background peripheral optical flow vehicle motion detector
│   ├── alerts.py               # Sound synthesizer and alarm loop manager
│   ├── system_health.py        # Laptop battery and camera connection health monitor
│   ├── shared_state.py         # Thread-safe singleton bridge between UI thread and Flask admin server
│   └── remote_admin.py         # Embedded Flask web server, Wi-Fi discovery, QR generator, & buzzer control API
├── storage/                    # Local SQLite database layer
│   ├── db.py                   # SQLite connection lifecycle, schema initialization, and auto-migrations
│   └── models.py               # Session, Event, and UserProfile dataclass CRUD operations
├── ui/                         # PyQt6 modern dark-themed GUI
│   ├── main_window.py          # Primary monitoring screen, video canvas, HUD, motion status, SharedState integration
│   ├── admin_dialog.py         # Supervisor PIN-gated alarm override modal (local)
│   ├── profile_dialog.py       # Persistent user calibration profile manager
│   ├── math_puzzle_window.py   # Cognitive alarm verification puzzle dialog
│   ├── settings_window.py      # Threshold, camera, motion, low-light, posture, and remote admin config dialog
│   ├── summary_window.py       # Historical session analytics with embedded matplotlib graphs
│   └── qr_dialog.py            # Wi-Fi & Mobile QR Code Access Dialog
├── tests/                      # Automated test suite (54/54 passing)
│   ├── test_classifier.py      # FSM state transition and hysteresis unit tests
│   ├── test_features.py        # EAR and MAR mathematical unit tests
│   ├── test_head_pose.py       # solvePnP Euler angle derivation and posture penalty tests
│   ├── test_math_puzzle.py     # Arithmetic puzzle generation and answer validation tests
│   ├── test_motion_detector.py # Vehicle motion detector and peripheral differencing tests
│   ├── test_multi_face.py      # Multi-face bounding box and driver ROI isolation tests
│   ├── test_preprocessing.py   # Brightness detection, CLAHE enhancement, and glare filter tests
│   ├── test_profiles.py        # SQLite UserProfile CRUD and calibration persistence tests
│   ├── test_yawn_frequency.py  # Rolling 5m yawn frequency escalation tests
│   ├── test_shared_state.py    # SharedState singleton, thread-safety, buzzer command, and event buffer tests
│   └── test_remote_admin.py    # Flask admin panel auth, status API, buzzer control, QR code, and event log tests
└── md_files/                   # System documentation & memories
    ├── README.md               # User-facing README with screenshot slots and quickstart guide
    ├── PRD.md                  # Product Requirements Document (v2.3)
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
  motion_threshold: 4.0
  window_size: 15
  min_moving_ratio: 0.4

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

Results: **54 passed in 2.58s (100% pass rate)**.
- `test_classifier.py` (4 tests): AWAKE state, prolonged eye closure, yawn increment, face loss recovery.
- `test_features.py` (4 tests): Euclidean distance, single eye EAR, empty landmarks fallback, empty mouth fallback.
- `test_head_pose.py` (2 tests): solvePnP fallback on empty landmarks, nodding & distraction posture evaluation.
- `test_math_puzzle.py` (2 tests): Arithmetic puzzle generation, answer verification.
- `test_motion_detector.py` (5 tests): Initialization, None/empty frame handling, static parked frames detection, moving scenery optical flow, reset function.
- `test_multi_face.py` (2 tests): Multi-face initialization, bounding box and center extraction.
- `test_preprocessing.py` (4 tests): Frame brightness calculation, CLAHE & gamma brightening, glare filter, integration test.
- `test_profiles.py` (1 test): UserProfile creation, retrieval, update, and deletion in SQLite.
- `test_yawn_frequency.py` (1 test): Rolling 5m window yawn accumulation and threshold escalation.
- `test_shared_state.py` (12 tests): Singleton pattern, metric read/write with motion metrics, concurrent thread-safety, buzzer command queue/consume, event buffer cap at 50, monitoring state tracking.
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
- [ ] **Milestone 15 (Future):** Pomodoro / Study interval timer widget.
- [ ] **Milestone 16 (Future):** Exportable PDF session reports using `reportlab`.
