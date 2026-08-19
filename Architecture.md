# System Architecture
## Drowsiness & Attention Detection System

---

## 1. High-Level Architecture

```
                       ┌───────────────────────────────────────────┐
                       │                Desktop App                 │
                       │                                             │
 ┌─────────────┐       │  ┌────────────┐   ┌───────────────────┐   │
 │   Webcam     │──────┼─▶│  Capture    │──▶│  Face/Landmark     │   │
 │  (Hardware)  │       │  │  Module     │   │  Detection Engine   │   │
 └─────────────┘       │  └────────────┘   └─────────┬──────────┘   │
                       │                              │              │
                       │                              ▼              │
                       │                    ┌────────────────────┐   │
                       │                    │  Feature Extractor  │   │
                       │                    │ (EAR, MAR, head pose)│  │
                       │                    └─────────┬──────────┘   │
                       │                              ▼              │
                       │                    ┌────────────────────┐   │
                       │                    │  State Classifier   │   │
                       │                    │ (thresholds + FSM)  │   │
                       │                    └─────────┬──────────┘   │
                       │                              ▼              │
                       │         ┌────────────────────┴──────────┐   │
                       │         ▼                                ▼  │
                       │  ┌─────────────┐                ┌──────────────┐
                       │  │ Alert Module │                │ Logging Module│
                       │  │ (audio/visual)│               │ (CSV/SQLite)  │
                       │  └─────────────┘                └──────────────┘
                       │         │                                │    │
                       │         ▼                                ▼    │
                       │  ┌──────────────────────────────────────────┐ │
                       │  │            UI Layer (PyQt/Tkinter)         │ │
                       │  │  Live feed | Status | Settings | Summary   │ │
                       │  └──────────────────────────────────────────┘ │
                       └───────────────────────────────────────────┘
```

## 2. Components

### 2.1 Capture Module
- Wraps `cv2.VideoCapture` to pull frames from the default/selected webcam.
- Runs on a dedicated thread to decouple frame acquisition from processing (avoids UI freeze).
- Handles camera reconnection and frame-drop recovery.

### 2.2 Face/Landmark Detection Engine
- Primary: **MediaPipe Face Mesh** (fast, CPU-friendly, 468 landmarks, no external model download issues).
- Fallback/alternative: **dlib 68-point landmark predictor** (heavier, but a well-known reference implementation).
- Outputs normalized landmark coordinates per frame.

### 2.3 Feature Extractor
- Computes:
  - **EAR (Eye Aspect Ratio)** from eye landmark subsets (left/right averaged).
  - **MAR (Mouth Aspect Ratio)** for yawn detection.
  - **Head pose (pitch/yaw)** — post-MVP — via solvePnP against a generic 3D face model.
- Applies temporal smoothing (rolling average over N frames) to reduce jitter/noise.

### 2.4 State Classifier
- Finite-state machine with states: `AWAKE → DROWSY_WARNING → DROWSY_ALERT`.
- Transitions driven by:
  - EAR below threshold for ≥ configurable consecutive frames → drowsy signal.
  - Blink-rate deviation over rolling window.
  - MAR above threshold → yawn event (contributes to fatigue score, not an alert by itself).
- Combines signals into a **fatigue score**; alert triggers once score crosses configured threshold.

### 2.5 Alert Module
- Visual: full-window banner/overlay color change (green → yellow → red).
- Audio: looping alarm sound via `playsound` or `simpleaudio`, stoppable by user action (keypress/click) to confirm alertness.
- Optional (future): system notification via OS-native APIs.

### 2.6 Logging Module
- Writes structured events to a local **SQLite** database (preferred over flat CSV for querying session history).
- Schema: `sessions`, `events` (timestamp, event_type, EAR value, MAR value, duration).
- No raw video/image frames are persisted by default (privacy-by-design).

### 2.7 UI Layer
- Built with **PyQt6** (richer widgets, better theming than Tkinter; justified given the app's dashboard needs).
- Screens:
  - **Live Monitor** — webcam feed with landmark overlay (toggleable), current status indicator.
  - **Settings** — threshold sliders, camera selection, alert sound selection.
  - **Session Summary** — post-session charts (drowsy events over time), exportable log.

## 3. Data Flow

1. Capture thread pushes frames into a thread-safe queue.
2. Processing thread pops frames, runs landmark detection, computes EAR/MAR.
3. Classifier updates FSM state per frame; smoothed over rolling window.
4. On state transition to `DROWSY_ALERT`, Alert Module fires; Logging Module records event.
5. UI thread polls shared state object (via Qt signals/slots) to update the live view without blocking capture/processing.

## 4. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.10+ | Ecosystem fit for CV/ML, matches existing skillset |
| Computer Vision | OpenCV, MediaPipe | Fast, CPU-friendly, no GPU dependency required |
| UI | PyQt6 | Native desktop feel, good widget/chart support |
| Storage | SQLite (via `sqlite3`) | Zero-config local persistence, queryable |
| Packaging | PyInstaller | Single-executable distribution for Windows/macOS/Linux |
| Testing | pytest | Unit tests for EAR/MAR math and FSM logic |
| Charts (summary) | `matplotlib` or `pyqtgraph` | Embed session charts directly in PyQt UI |

## 5. Threading Model

- **Thread 1 — Capture:** continuously reads frames, minimal processing, pushes to queue.
- **Thread 2 — Processing:** landmark detection + feature extraction + classification (CPU-bound; releases GIL during OpenCV/MediaPipe native calls).
- **Main Thread — UI (Qt event loop):** renders frames/state, handles user input, never blocks on CV work.
- Communication via `queue.Queue` (frames) and Qt `pyqtSignal` (state updates) to avoid race conditions.

## 6. Deployment Model

- Distributed as a packaged executable (PyInstaller) for non-technical end users (students).
- Developer/technical users can run directly via `pip install -r requirements.txt && python main.py`.
- Config stored in a local `config.yaml` (thresholds, camera index, alert preferences) — editable without touching code.

## 7. Future Extensibility

- Swap local SQLite logging for an optional REST sync to a supervisor dashboard (v2, opt-in only, requires explicit consent flow given privacy sensitivity).
- Add head-pose-based nod detection as a third fatigue signal.
- Plug-in architecture for detection "signals" so new fatigue indicators can be added without touching the classifier core.
