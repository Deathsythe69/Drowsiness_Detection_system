# Product Requirements Document (PRD)
## Drowsiness & Attention Detection System

**Version:** 1.0
**Platform:** Desktop Application (Python + OpenCV)
**Use Case:** Workplace / Student Attention Monitoring
**Author:** Debasis
**Date:** August 2026

---

## 1. Overview

A desktop application that uses a standard webcam to monitor a user's eyes and facial state in real time, detect signs of drowsiness or sustained inattention, and alert the user (or a supervisor/dashboard) before performance or safety is compromised. Designed for use in workplaces (control rooms, remote work monitoring, long computer sessions) and student settings (study sessions, online exam proctoring, e-learning engagement tracking).

## 2. Problem Statement

Extended screen time — during work shifts, remote study, or online exams — leads to fatigue-driven lapses in attention. These lapses reduce productivity, learning retention, and in supervised settings (exam proctoring, safety-critical monitoring) can enable rule violations or missed safety events. There is no lightweight, privacy-respecting, locally-run tool that gives real-time feedback on drowsiness without depending on cloud services or specialized hardware.

## 3. Goals

- Detect drowsiness (prolonged eye closure, reduced blink rate, yawning, head-nodding) using only a standard webcam.
- Alert the user in real time (audio/visual) with minimal false positives.
- Log session-level attention data for later review (individual self-review or supervisor dashboard).
- Run entirely on-device by default — no mandatory cloud dependency — for privacy in student/workplace settings.
- Be lightweight enough to run alongside normal work/study applications without noticeable lag.

## 4. Non-Goals (Out of Scope for v1)

- Driver monitoring / in-vehicle deployment (different hardware, lighting, and regulatory constraints).
- Mobile app version.
- Multi-camera or multi-person simultaneous monitoring.
- Emotion recognition or productivity scoring beyond attention/drowsiness.
- Cloud-based analytics dashboard (may be a v2 consideration).

## 5. Target Users

| User Type | Context | Needs |
|---|---|---|
| Student | Self-study, online exams | Self-alerts, session summary, exam-proctoring mode |
| Remote employee | Long work sessions | Break reminders, personal focus analytics |
| Supervisor/Institution (optional) | Proctored exams, monitored workstations | Session logs/reports, configurable thresholds |

## 6. Key Features

### 6.1 MVP (v1.0)
1. **Real-time face & eye tracking** via webcam feed.
2. **Eye Aspect Ratio (EAR)-based drowsiness detection** — flags prolonged eye closure beyond a configurable threshold/duration.
3. **Blink rate monitoring** — abnormally low blink rate as a secondary signal.
4. **Yawn detection** via Mouth Aspect Ratio (MAR).
5. **Real-time alerts** — on-screen banner + audio alarm when drowsiness is detected.
6. **Session logging** — timestamped events (drowsy, alert, awake) written locally (CSV/SQLite).
7. **Configurable sensitivity** — thresholds adjustable via a settings panel or config file.
8. **Basic dashboard/summary screen** — post-session view of drowsiness events, total alert count, session duration.

### 6.2 Post-MVP (v1.1+)
- Head-pose estimation (nodding-off detection) as a third signal.
- Exam/study "focus mode" with break-reminder scheduling (Pomodoro-style).
- Exportable PDF/CSV session reports.
- Optional multi-user profile support (per-user calibration).
- Optional lightweight local dashboard (Flask/Streamlit) for supervisors reviewing multiple session logs.

## 7. Success Metrics

- **Detection accuracy:** ≥90% true positive rate on prolonged eye-closure events in varied lighting, validated against manually labeled test clips.
- **False positive rate:** <10% (avoid alert fatigue).
- **Latency:** Alert triggered within 1 second of sustained drowsy state.
- **Performance:** Runs at ≥15 FPS on a mid-range laptop (integrated GPU / CPU-only).
- **Adoption proxy:** Usable in a 2-hour continuous session without crashes or memory leaks.

## 8. Constraints & Assumptions

- Assumes a single, front-facing, reasonably well-lit user per session.
- Requires a functioning webcam (built-in or USB).
- No internet connection required for core functionality.
- Python 3.10+ target environment; cross-platform (Windows/Linux/macOS) where camera drivers permit.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Poor lighting reduces detection accuracy | Add brightness/contrast pre-processing; recommend lighting setup guide |
| Glasses/reflections interfere with eye landmarks | Test dataset should include glasses-wearing subjects; consider IR fallback in future |
| False alerts cause user annoyance | Tunable thresholds + temporal smoothing (require N consecutive drowsy frames) |
| Privacy concerns (webcam monitoring) | No footage stored by default; only derived metrics logged; explicit on-screen recording indicator |

## 10. Open Questions

- Should exam-proctoring mode capture screenshots/evidence, or purely behavioral flags? (Privacy/compliance implications.)
- Is a supervisor-facing report needed for v1, or is this purely self-monitoring initially?
