# Design Document
## Drowsiness & Attention Detection System

---

## 1. Purpose

This document translates the Architecture into concrete module-, class-, and algorithm-level design: data structures, key formulas, UI layout, and configuration schema.

## 2. Project Structure

```
drowsiness-detector/
├── main.py                  # Entry point, launches Qt app
├── config.yaml               # User-editable thresholds & settings
├── requirements.txt
├── core/
│   ├── capture.py            # Webcam capture thread
│   ├── landmarks.py          # MediaPipe/dlib wrapper
│   ├── features.py           # EAR, MAR, head-pose calculations
│   ├── classifier.py         # FSM / fatigue scoring logic
│   └── alerts.py             # Audio/visual alert triggers
├── storage/
│   ├── db.py                 # SQLite connection & schema
│   └── models.py             # Session/Event data classes
├── ui/
│   ├── main_window.py        # Live Monitor screen
│   ├── settings_window.py    # Settings screen
│   └── summary_window.py     # Session summary/charts
├── assets/
│   └── alarm.wav
└── tests/
    ├── test_features.py
    └── test_classifier.py
```

## 3. Core Algorithms

### 3.1 Eye Aspect Ratio (EAR)

For six eye landmarks `p1..p6` (per eye):

```
EAR = (‖p2 - p6‖ + ‖p3 - p5‖) / (2 * ‖p1 - p4‖)
```

- Average EAR across both eyes each frame.
- EAR drops sharply toward 0 during eye closure, stays roughly stable (~0.25–0.3) when open (subject-dependent; calibrated per session).

**Drowsy condition:** `EAR < EAR_THRESHOLD` for `CONSEC_FRAMES` consecutive frames (default: threshold 0.21, ~20 consecutive frames at 15–20 FPS ≈ 1 second).

### 3.2 Mouth Aspect Ratio (MAR) — Yawn Detection

```
MAR = (‖p2 - p8‖ + ‖p3 - p7‖ + ‖p4 - p6‖) / (2 * ‖p1 - p5‖)
```

- `MAR > MAR_THRESHOLD` sustained for a shorter window (e.g., 15 frames) flags a yawn event.
- Yawn events increment a rolling "fatigue score" rather than triggering an alert directly (yawning alone is not conclusive drowsiness).

### 3.3 Fatigue Score & FSM

- Rolling window (e.g., last 60 seconds) tracks: drowsy-eye events, yawn count, blink-rate deviation.
- Score weighting (tunable): eye closure events weighted highest, yawns moderate, blink-rate lowest.
- **States:**
  - `AWAKE` — score below warning threshold.
  - `DROWSY_WARNING` — score crosses warning threshold → visual cue only (yellow banner).
  - `DROWSY_ALERT` — score crosses alert threshold, or single sustained eye-closure ≥1s → audio + visual alert.
- State resets to `AWAKE` after a period of normal EAR/blink behavior (hysteresis to avoid alert flapping).

### 3.4 Calibration (First 10–15 Seconds of Session)

- Capture baseline EAR/MAR for the specific user/lighting condition.
- Adjust thresholds ± a tolerance band around the observed baseline rather than using a single fixed global threshold — improves accuracy across different eye shapes/glasses.

## 4. Data Model

### `sessions` table
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| user_label | TEXT | optional, self-identified |
| start_time | DATETIME | |
| end_time | DATETIME | |
| baseline_ear | REAL | from calibration |

### `events` table
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| session_id | INTEGER FK | |
| timestamp | DATETIME | |
| event_type | TEXT | `drowsy_warning`, `drowsy_alert`, `yawn`, `blink` |
| ear_value | REAL | |
| mar_value | REAL | |

## 5. UI Design

### 5.1 Live Monitor Screen
- Left: live webcam feed with optional landmark overlay (toggle).
- Right panel:
  - Status indicator (green/yellow/red circle + label).
  - Current EAR/MAR readout (numeric, small font — for power users/debugging).
  - Session timer.
  - "Dismiss Alert" button (stops alarm, logs user acknowledgment).

### 5.2 Settings Screen
- Sliders: EAR threshold, consecutive-frame count, MAR threshold, alert volume.
- Dropdown: camera source (if multiple).
- Checkbox: "Show landmark overlay", "Enable yawn detection".
- Save/Reset to defaults buttons; persists to `config.yaml`.

### 5.3 Session Summary Screen
- Line chart: EAR value over session duration, with alert events marked.
- Stat cards: total drowsy alerts, total yawns, longest continuous-focus streak.
- Export buttons: CSV export, PDF summary (post-MVP).

## 6. Configuration Schema (`config.yaml`)

```yaml
camera:
  index: 0
thresholds:
  ear_threshold: 0.21
  ear_consec_frames: 20
  mar_threshold: 0.6
  mar_consec_frames: 15
  fatigue_warning_score: 40
  fatigue_alert_score: 75
alerts:
  sound_file: "assets/alarm.wav"
  volume: 0.8
ui:
  show_landmarks: true
  theme: "dark"
```

## 7. Error Handling & Edge Cases

- **No face detected:** after 3 seconds with no landmarks, show "Face not detected" status (distinct from drowsy) — do not fire drowsiness alert.
- **Multiple faces detected:** track the largest/most central bounding box only; ignore background faces.
- **Camera disconnect:** capture thread attempts reconnection every 2 seconds, UI shows a non-blocking warning banner.
- **Low-light conditions:** apply CLAHE (adaptive histogram equalization) to the frame before landmark detection to improve robustness.

## 8. Testing Strategy

- Unit tests for EAR/MAR math against known landmark fixtures with expected output ranges.
- FSM unit tests: simulate sequences of EAR values, assert correct state transitions and hysteresis behavior.
- Manual test matrix: varied lighting (bright/dim), with/without glasses, varied distance from camera.
