# 📊 Dataset Sourcing, Benchmarking Plan & Evaluation Strategy

This document outlines suitable public datasets, licensing, fine-tuning protocols, and testing frameworks for evaluating the **Drowsiness & Attention Detection System** under complex conditions (low light, glasses glare, multi-occupant cabins, and head nodding).

---

## 1. Candidate Dataset Identification & Licensing Review

| Dataset | Modality & Size | Target Scenarios | Annotation Details | Licensing & Access | Suitability for This Project |
|---|---|---|---|---|---|
| **NTHU-DDD** (*National Tsing Hua University*) | 36 subjects, ~9.5 hours video, IR & RGB | Daytime, Nighttime (Low light), Glasses, Sunglasses, Yawning, Nodding | Frame-by-frame binary & multi-class drowsiness labels | Free for Non-Commercial Academic/Research upon agreement | **Very High** (Directly evaluates low-light, glasses glare, and head nodding) |
| **YawDD** (*University of Ottawa*) | 107 subjects, 340+ video clips (dashcam & mirror angles) | Normal talking, singing, sustained yawning, head movements | Video & frame-level yawning and facial bounding boxes | Free for Academic Research upon email request | **High** (Ideal for calibrating rolling yawn frequency and MAR thresholds) |
| **UTA-RLDD** (*Univ. of Texas at Arlington*) | 60 subjects, 180 multi-stage videos (~30 hrs) | Real-life fatigue progression (Alert $\rightarrow$ Low Vigilance $\rightarrow$ Drowsy) | Multi-stage self-reported & observer ground truth | Creative Commons / Academic Use | **High** (Tests long-duration session attention curves & slow eye closure) |
| **CEW** (*Closed Eyes in the Wild*) | 4,846 annotated facial crops | Extreme lighting, varying head poses, sunglasses, diverse ethnicities | Binary Open/Closed eye ground truth | Open Public Academic License | **High** (Benchmark for static EAR threshold ROC curves) |
| **DMD** (*Driver Monitoring Dataset*) | Multi-camera in-vehicle video suite | Driver vs Passenger multi-person cabin tracking, distraction, gaze, drowsiness | Spatio-temporal cabin annotations | Academic Research License | **Medium-High** (Validates driver ROI isolation in multi-person cabins) |

---

## 2. Threshold Calibration & Fine-Tuning Methodology

### 2.1 Optimization Objective
To minimize false alarms while maintaining $\ge 90\%$ detection rate within 1.0 second of true microsleeps:

$$\max_{\theta_{\text{EAR}}, \theta_{\text{MAR}}, \theta_{\text{pitch}}} F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Subject to:
- $\text{False Positive Rate (FPR)} < 5\%$ on active baseline study/driving footage.
- $\text{True Positive Rate (TPR)} \ge 90\%$ on prolonged closure clips ($\ge 1.0\text{s}$).
- Detection Latency $\le 1.0\text{s}$.

---

### 2.2 Benchmarking Matrix Across Test Conditions

```
                                 PERFORMANCE TARGET MATRIX
┌───────────────────────────┬──────────────┬──────────────┬──────────────────┬──────────────┐
│ Scenario / Condition      │ Min. TPR (%) │ Max. FPR (%) │ Max Latency (ms) │ Min. FPS     │
├───────────────────────────┼──────────────┼──────────────┼──────────────────┼──────────────┤
│ 1. Well-lit Normal        │   ≥ 95%      │    < 3%      │     800 ms       │   ≥ 25 FPS   │
│ 2. Low-Light (CLAHE Mode) │   ≥ 90%      │    < 7%      │    1000 ms       │   ≥ 20 FPS   │
│ 3. Eyeglasses with Glare  │   ≥ 88%      │    < 8%      │    1000 ms       │   ≥ 20 FPS   │
│ 4. Multi-Person Cabin     │   ≥ 92%      │    < 5%      │     900 ms       │   ≥ 18 FPS   │
│ 5. Head Nodding / Droop   │   ≥ 90%      │    < 6%      │    1000 ms       │   ≥ 20 FPS   │
└───────────────────────────┴──────────────┴──────────────┴──────────────────┴──────────────┘
```

---

## 3. Automated Validation Framework (`tests/`)

To validate detection accuracy continuously without requiring live human testers during automated builds:

### 3.1 Synthetic & Landmark-Fixture Test Suites
1. **`tests/test_preprocessing.py`**:
   - Synthesizes synthetic dark frames ($\text{mean luminance} < 30.0$) and checks that CLAHE/Gamma lifts contrast and restores detectable edges.
   - Tests specular glare suppression filters on synthetic white saturated reflection patches.
2. **`tests/test_head_pose.py`**:
   - Injects canonical 3D facial landmark rotations.
   - Verifies `solvePnP` recovers negative pitch (nodding) and positive/negative yaw (distraction) accurately within $\pm 2^\circ$.
3. **`tests/test_yawn_frequency.py`**:
   - Simulates sequences of MAR frames over a mock 5-minute timeline.
   - Asserts that 1 yawn adds fatigue, 2 yawns warn, and $\ge 3$ yawns escalate the state to `DROWSY_ALERT`.
4. **`tests/test_profiles.py`**:
   - Tests saving, loading, updating, and deleting persistent calibration profiles in SQLite.
5. **`tests/test_multi_face.py`**:
   - Feeds multiple bounding boxes simulating driver (center ROI) and passenger (side ROI) and verifies driver priority.
