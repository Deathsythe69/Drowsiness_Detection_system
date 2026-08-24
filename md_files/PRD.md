# Product Requirements Document (PRD)
## Drowsiness & Attention Detection System

**Version:** 2.3 (Wi-Fi / Hotspot Mobile QR Access & Vehicle Motion Gating)  
**Platform:** Desktop Application (Python, OpenCV, MediaPipe, PyQt6)  
**Use Case:** Workplace, Online Learning, Exam Proctoring & Driver Inattention Monitoring  
**Author:** Debasis Panigrahi ([@Deathsythe69](https://github.com/Deathsythe69))  
**Date:** August 2026  

---

## 1. Overview & Problem Statement

Extended screen time and prolonged vigilance tasks lead to fatigue-driven lapses in cognitive attention. This system provides a **100% on-device, privacy-preserving desktop application** that monitors eye closure dynamics, blinking patterns, yawning frequency, and posture/head-nodding in real time to alert the user or supervisor before performance or safety is compromised.

In automotive and shared cabin scenarios:
1. Alerting must be strictly focused on the **primary driver** while actively driving.
2. Alarms are suppressed when the vehicle is stationary or parked.
3. Supervisors and co-passengers can access the control panel over **both Wi-Fi / Hotspot and wired LAN**, with instant camera QR-code scanning on mobile phones.

---

## 2. Key Features (v2.3)

### 2.1 Core Detection & Robustness
1. **Adaptive Low-Light Preprocessing:** Detects low ambient brightness ($L < 65$) and dynamically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gamma correction ($\gamma = 0.6$) with specular glare suppression on eyeglasses.
2. **Multi-Person Detection & Driver-Only Alerting:** Tracks multiple cabin faces via MediaPipe Face Mesh (`max_num_faces=4`) but selectively isolates and alerts only the primary driver within the central/driver Region of Interest (ROI). Passengers are rendered with non-intrusive bounding boxes and explicitly excluded from all alert evaluations.
3. **Vehicle Motion Detection & Driving Gating:** Uses background optical flow / peripheral frame differencing to determine if the vehicle is in motion. Audible buzzer alarms are gated to trigger **only when the driver is drowsy while the vehicle is actively moving**. If the car is parked or stopped, the audible alarm is muted to prevent disturbance during rest stops.
4. **Personalized Calibration & Persistent User Profiles:** Records baseline EAR, blink frequency, and variance during a 10s calibration routine. Stores named user profiles in SQLite so returning users can load customized thresholds instantly.
5. **Rolling-Window Yawn Frequency Escalation:** Tracks yawning occurrences across a rolling 5-minute (300s) window. If $\ge 3$ yawns occur within the window, the system triggers a frequency escalation alert.
6. **Head Pose & Posture Estimation:** Uses 3D facial canonical model mapping via `solvePnP` to calculate real-time Euler angles (Pitch, Yaw, Roll), penalizing head drooping (nodding off) and sustained distraction.
7. **Device Battery & Graceful Camera Disconnect Resilience:** Monitors laptop battery state via `psutil`. Alerts users when battery drops below 20%. Automatically flushes and saves session logs if the camera disconnects.
8. **Dual-Tier Alarm Dismissal & Supervisor PIN Override:**
   - **User Dismissal:** Requires solving an interactive cognitive arithmetic puzzle.
   - **Admin Override:** PIN-authenticated modal (`"1234"`) for supervisors to immediately silence alarms with persistent audit logging.
9. **Wi-Fi & Mobile QR Code Remote Admin Panel:**
   - Embedded Flask web server accessible across **Wi-Fi, Mobile Hotspot, and LAN**.
   - Auto-discovers Wi-Fi network interfaces and generates a **scannable QR code** directly on the desktop screen (`WiFiAccessDialog`) and via `/qr`.
   - Any smartphone connected to the same Wi-Fi / hotspot can point its camera at the screen to load the PIN-authenticated admin dashboard in seconds.
   - Allows supervisors to monitor live telemetry, vehicle motion state, mute/unmute the buzzer, and inspect the event audit log.
10. **Session Telemetry & Post-Session Analytics:** Comprehensive SQLite database recording sessions, events, and metrics with visual matplotlib summary dashboards.

---

## 3. Scope Boundaries & Assumptions

- **Primary Occupant Scope:** Detection and alert logic targets the primary driver inside the active ROI. Other cabin passengers/bystanders are explicitly ignored for alarm triggering.
- **Motion Gating:** Auditory alerts require active vehicle motion when `motion.require_motion_for_alert` is enabled.
- **Network Compatibility:** Remote admin panel is compatible with Wi-Fi, Ethernet LAN, and mobile vehicle hotspots on the same subnet.
- **Privacy Assurance:** Zero raw video frames or photos are saved to disk or network. Only derived numeric metrics (EAR, MAR, Euler angles, timestamps, motion scores) are logged.
- **Hardware Requirement:** Standard USB or built-in webcam running on consumer CPU hardware (≥20 FPS).

---

## 4. Success Metrics

- **True Positive Rate:** $\ge 90\%$ on prolonged eye closure ($\ge 1.0\text{s}$) across daytime, low-light, and glasses-wearing scenarios while driving.
- **False Alarm Rate:** $< 5\%$ during normal alertness; $0\%$ alarm disturbance when parked.
- **Alert Latency:** Triggered within $1.0\text{s}$ of sustained fatigue condition during motion.
- **Mobile QR Pairing Latency:** Instant connection upon camera scan on Wi-Fi / Hotspot.
