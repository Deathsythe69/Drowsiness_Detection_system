# Product Requirements Document (PRD)
## Drowsiness & Attention Detection System

**Version:** 2.5 (Intelligent Eyewear Classification, Through-Reflection Vision & Sunglasses Surrogate Mode)  
**Platform:** Desktop Application (Python, OpenCV, MediaPipe, PyQt6)  
**Use Case:** Workplace, Online Learning, Exam Proctoring & Automotive Driver Inattention Monitoring  
**Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))  
**Date:** August 2026  

---

## 1. Overview & Problem Statement

Extended screen time and prolonged vigilance tasks lead to fatigue-driven lapses in cognitive attention. This system provides a **100% on-device, privacy-preserving desktop application** that monitors eye closure kinetics, PERCLOS, eyewear states (bare eyes, regular glasses with glare, dark sunglasses), blinking patterns, yawning frequency, and posture/head-nodding in real time to alert the user or supervisor before performance or safety is compromised.

In automotive and shared cabin scenarios:
1. Alerting must be strictly focused on the **primary driver** while actively driving.
2. Alarms are suppressed when the vehicle is stationary or parked.
3. Early fatigue is caught via **automotive PERCLOS** before full microsleeps occur.
4. **Regular Glasses**: System dampens lens reflection glints and sees through the glass to accurately track eyelids and pupils.
5. **Sunglasses**: Bypasses occluded eye closure to eliminate dark-lens false alarms, switching to surrogate physiological signals (**MAR yawn kinetics & solvePnP 3D Head Pose**).
6. Video pipeline maintains zero lag ($<1.0\text{ms}$ neural inference, asynchronous disk I/O, and zero UI thread stalls).
7. Critical drowsiness events are automatically recorded to a **rolling blackbox evidence buffer**.
8. Supervisors and co-passengers can access the control panel over **both Wi-Fi / Hotspot and wired LAN**, with instant camera QR-code scanning on mobile phones.

---

## 2. Key Features (v2.5)

### 2.1 Core Detection & Robustness
1. **Intelligent Eyewear Classification:** Multi-feature classifier detecting `NONE` (bare eyes), `REGULAR_GLASSES` (clear/prescription), and `SUNGLASSES` (dark/tinted lenses) with ocular-to-skin luminance ratio and 15-frame hysteresis voting.
2. **Through-Reflection Clear Glasses Vision:** Specialized anti-glare specular highlight filtering and localized CLAHE that penetrates lens reflections to track pupil/eyelid movements clearly.
3. **Adaptive Sunglasses Surrogate Fatigue Mode:** Automatically bypasses eye closure alerts when dark sunglasses are worn, relying on high-sensitivity surrogate physiological tracking via **Mouth Aspect Ratio (MAR yawn kinetics)** and **solvePnP 3D Head Pose (Pitch downward nodding & Yaw inattention/wobble)**.
4. **Automotive-Standard PERCLOS ($P_{80}$):** Measures Percentage of Eye Closure across rolling 60-frame (~2s) and 300-frame (~10s) windows to reliably detect slow eyelid droop, heavy blinking, and fatigue accumulation before acute microsleep.
5. **Vectorized Low-Light Eye CNN (`im2col` + BLAS GEMM):** Fully vectorized pure NumPy convolutional neural network executing in **<0.9ms** on standard CPU, evaluating dual-eye openness without PyTorch/TensorFlow weight.
6. **Adaptive Low-Light Preprocessing:** Detects low ambient brightness ($L < 65$) and dynamically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gamma correction ($\gamma = 0.6$) with specular glare suppression on eyeglasses.
7. **Zero-Latency Async SQLite Worker (`AsyncDBLogger`):** Background queue worker completely decouples disk writes from the Qt GUI thread, ensuring 0ms disk delay and 60 FPS smooth video.
8. **Continuous Active Learning & Live Trainer:** Automatically harvests categorized edge-case eye crops during driving alerts and enables 1-click model self-training with immediate in-memory weight hot-reloading.
9. **Automated Blackbox Video Evidence Recorder:** Circular pre/post rolling frame buffer automatically records and persists high-resolution MP4 video clips whenever a critical drowsiness alert is triggered.
10. **Multi-Person Detection & Driver-Only Alerting:** Tracks multiple cabin faces via MediaPipe Face Mesh (`max_num_faces=4`) but selectively isolates and alerts only the primary driver within the central/driver Region of Interest (ROI). Passengers are rendered with non-intrusive bounding boxes and explicitly excluded from all alert evaluations.
11. **Vehicle Motion Detection & Driving Gating:** Uses background optical flow / peripheral frame differencing to determine if the vehicle is in motion. Audible buzzer alarms are gated to trigger **only when the driver is drowsy while the vehicle is actively moving**. If the car is parked or stopped, the audible alarm is muted to prevent disturbance during rest stops.
12. **Personalized Calibration & Persistent User Profiles:** Records baseline EAR, blink frequency, and variance during a 10s calibration routine. Stores named user profiles in SQLite so returning users can load customized thresholds instantly.
13. **Rolling-Window Yawn Frequency Escalation:** Tracks yawning occurrences across a rolling 5-minute (300s) window. If $\ge 3$ yawns occur within the window, the system triggers a frequency escalation alert.
14. **Head Pose & Posture Estimation:** Uses 3D facial canonical model mapping via `solvePnP` to calculate real-time Euler angles (Pitch, Yaw, Roll), penalizing head drooping (nodding off) and sustained distraction.
15. **Device Battery & Graceful Camera Disconnect Resilience:** Monitors laptop battery state via `psutil`. Alerts users when battery drops below 20%. Automatically flushes and saves session logs if the camera disconnects.
16. **Dual-Tier Alarm Dismissal & Supervisor PIN Override:**
   - **User Dismissal:** Requires solving an interactive cognitive arithmetic puzzle.
   - **Admin Override:** PIN-authenticated modal (`"1234"`) for supervisors to immediately silence alarms with persistent audit logging.
17. **Wi-Fi & Mobile QR Code Remote Admin Panel:**
   - Embedded Flask web server accessible across **Wi-Fi, Mobile Hotspot, and LAN**.
   - Auto-discovers Wi-Fi network interfaces and generates a **scannable QR code** directly on the desktop screen (`WiFiAccessDialog`) and via `/qr`.
   - Any smartphone connected to the same Wi-Fi / hotspot can point its camera at the screen to load the PIN-authenticated admin dashboard in seconds.
   - Allows supervisors to monitor live telemetry, vehicle motion state, mute/unmute the buzzer, and inspect the event audit log.
18. **Session Telemetry & Post-Session Analytics:** Comprehensive SQLite database recording sessions, events, and metrics with visual matplotlib summary dashboards.

---

## 3. Scope Boundaries & Assumptions

- **Primary Occupant Scope:** Detection and alert logic targets the primary driver inside the active ROI. Other cabin passengers/bystanders are explicitly ignored for alarm triggering.
- **Motion Gating:** Auditory alerts require active vehicle motion when `motion.require_motion_for_alert` is enabled.
- **Network Compatibility:** Remote admin panel is compatible with Wi-Fi, Ethernet LAN, and mobile vehicle hotspots on the same subnet.
- **Hardware Requirement:** Standard USB or built-in webcam running on consumer CPU hardware ($\ge 30$ FPS).

---

## 4. Success Metrics

- **True Positive Rate:** $\ge 95\%$ on prolonged eye closure ($\ge 1.0\text{s}$) and slow PERCLOS fatigue across daytime, low-light, and glasses-wearing scenarios while driving.
- **False Alarm Rate:** $< 3\%$ during normal alertness; $0\%$ alarm disturbance when parked.
- **Frame Latency:** Under $30\text{ms}$ total per-frame processing latency (neural inference $<1.0\text{ms}$).
- **Video Playback Smoothness:** 0 dropped UI frames, 0 mid-stream video freezes.
