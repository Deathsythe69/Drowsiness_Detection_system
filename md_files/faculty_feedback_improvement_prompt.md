# Prompt: Improve Drowsiness Detection System per Faculty Feedback

Repo: https://github.com/Deathsythe69/Drowsiness_Detection_system

I received the following faculty feedback on my drowsiness detection system (Python, OpenCV, MediaPipe, PyQt6 — repo structure: `core/`, `ui/`, `storage/`, `tests/`, `main.py`, `config.yaml`). Please review the existing code in `core/` and `ui/` first, then implement the following improvements. Work through them one at a time, explain each change, and update `config.yaml` for any new tunable parameters rather than hardcoding values.

## 1. Low-light robustness
- Detect when ambient/frame brightness is too low for reliable landmark detection (e.g. mean pixel intensity below a threshold).
- Apply preprocessing (histogram equalization / CLAHE, gamma correction) before passing frames to MediaPipe when brightness is low.
- Detect and reduce the impact of light reflection/glare (e.g. off glasses) so it isn't misread as an eye-open/closed signal.
- Show a UI indicator when the system is operating in a degraded-lighting state.

## 2. Person-specific eye blink calibration
- Blink frequency and baseline EAR vary between individuals — add a short calibration step at session start that records a user's normal blink rate and EAR range.
- Use this calibration to set personalized thresholds instead of one fixed global EAR threshold.
- Store calibration profiles (per user, if feasible) in `storage/` so returning users don't have to recalibrate every session.

## 3. Yawn frequency tracking
- Track yawn count and timestamps over a rolling time window (not just single-yawn detection), since frequency matters more than one occurrence.
- Escalate the alert level if yawn frequency crosses a threshold within a set period (e.g. 3+ yawns in 5 minutes).

## 4. Body-language / posture signals ("lazy person" characteristics)
- Extend detection beyond the face to include basic posture cues — slouching, head tilt/droop angle, reduced movement over time — using MediaPipe Pose or landmark-based head-pose estimation.
- Combine posture signal with EAR/MAR as a secondary signal to the drowsiness classifier, not a standalone trigger.

## 5. Camera / device battery handling
- If running on a device that exposes battery status (e.g. laptop webcam use case), detect low battery and warn the user that detection reliability may drop if the system shuts down mid-session.
- Ensure the app fails gracefully (saves session log, shows a clear warning) rather than crashing if the camera disconnects due to power loss.

## 6. Admin control to silence/override the buzzer
- Add an admin-authenticated control (simple PIN/password gate is fine) to mute or stop the active buzzer alert without ending the monitoring session.
- Log every manual override event (who, when) to the session log for accountability.
- Keep this separate from the normal user-facing start/stop/pause controls in `ui/`.

## 7. Multi-person detection and driver identification
- Detect multiple faces in frame and distinguish the driver from passengers — use face position/size (driver typically closest to camera / in a defined region of interest) rather than assuming a single face.
- Only run drowsiness alerting logic on the identified driver; passengers should not trigger false alerts.
- Note: passengers who are sleeping in other seats are out of scope for this alert — confirm the ROI logic explicitly excludes them rather than silently missing them, and document this as a known scope boundary.

## 8. Dataset sourcing and training
- Identify and shortlist existing public datasets suitable for these scenarios (e.g. driver drowsiness/yawning datasets, low-light face datasets, multi-face-in-vehicle datasets). Summarize licensing and suitability for each candidate.
- Propose a plan for fine-tuning or threshold-tuning against the chosen dataset(s), and outline what a `tests/` validation set should look like to measure accuracy/false-positive rate before and after training.

## Process notes
- Implement and test each numbered item independently so it can be demoed/reviewed separately.
- Update the PRD/architecture docs to reflect these additions once implemented.
- Flag any item where the existing architecture would need a significant restructure, rather than forcing a fit.
