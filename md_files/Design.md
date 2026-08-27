# Design Document
## Drowsiness & Attention Detection System (v2.3)

---

## 1. Mathematical Models & Formulas

### 1.1 Eye Aspect Ratio (EAR) & Automotive PERCLOS
For 6 landmarks per eye:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- **Default Baseline:** $\approx 0.25 - 0.35$
- **Alert Threshold:** Personal threshold from calibration (default $\approx 70\%$ of open baseline).

#### Automotive Standard PERCLOS ($P_{80}$)
PERCLOS measures the proportion of time that the driver's eyes are at least 80% closed over a designated observation interval $W$ (e.g. $W=60$ frames $\approx 2$s and $W=300$ frames $\approx 10$s):
$$\text{PERCLOS}_W = \frac{1}{W} \sum_{k=0}^{W-1} \mathbb{I}\left(\text{EAR}_{t-k} < \tau_{\text{closed}} \lor P(\text{Eye Closed})_{t-k} \ge 0.50\right)$$
- **Warning Condition:** $\text{PERCLOS}_{60} \ge 0.35 \implies \text{Fatigue score accumulates by } 2.5\times$
- **Normal Blink Filter:** Brief reflexive eyelid closures where duration $D < 15 \text{ frames}$ are filtered out from triggering acute alerts.

### 1.2 Vectorized Neural Inference via `im2col` + BLAS GEMM
Given input feature map $X \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ and convolution kernel $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$:
1. **Window Vectorization (`im2col`):**
   $$X_{\text{col}} = \text{sliding\_window\_view}(X, (k_h, k_w)) \in \mathbb{R}^{B \cdot H_{\text{out}} \cdot W_{\text{out}} \times (C_{\text{in}} \cdot k_h \cdot k_w)}$$
2. **BLAS Matrix Multiplication (GEMM):**
   $$Y_{\text{col}} = X_{\text{col}} \cdot K_{\text{flat}}^T + b \quad \in \mathbb{R}^{B \cdot H_{\text{out}} \cdot W_{\text{out}} \times C_{\text{out}}}$$
3. **Reshape & Activation:**
   $$Y = \text{ReLU}\left(\text{Reshape}(Y_{\text{col}})\right)$$
This matrix formulation executes on CPU in $<0.9\text{ms}$ per dual-eye batch.

### 1.3 Mouth Aspect Ratio (MAR) & Yawn Frequency
$$\text{MAR} = \frac{\|p_2 - p_8\| + \|p_3 - p_7\| + \|p_4 - p_6\|}{2 \cdot \|p_1 - p_5\|}$$
- **Yawn Condition:** $\text{MAR} > 0.60$ for $\ge 15$ consecutive frames.
- **Rolling Window Escalation:**
  $$N_{\text{yawns}}(t, \Delta t = 300\text{s}) \ge 3 \implies \text{Escalate Alert State}$$

### 1.4 Head Pose via 3D Perspective-n-Point (solvePnP)
- Canonical 3D facial anchors:
  - Nose tip (`1`): $(0, 0, 0)$
  - Chin (`152`): $(0, -330, -65)$
  - Left eye outer (`33`): $(-225, 170, -135)$
  - Right eye outer (`263`): $(225, 170, -135)$
  - Left mouth (`61`): $(-150, -150, -125)$
  - Right mouth (`291`): $(150, -150, -125)$
- Derived Euler angles:
  - **Pitch $< -18^\circ$**: Head drooping down (microsleep nod).
  - **$|\text{Yaw}| > 25^\circ$**: Distraction / Looking away.

### 1.5 Vehicle Motion Optical Flow Estimation
For consecutive grayscale frames $I_{t-1}, I_t$ smoothed with Gaussian kernel $G_{\sigma=5}$:
$$\Delta I = |G(I_t) - G(I_{t-1})|$$
$$\text{Motion Score} = \frac{1}{|M_{\text{periph}}|} \sum_{(x, y) \in M_{\text{periph}}} \Delta I(x, y)$$
where $M_{\text{periph}}$ is the peripheral mask covering top, bottom, and side window strips.
$$\text{Frame Moving} = \mathbb{I}(\text{Motion Score} > \tau_{\text{motion}})$$
$$\text{Moving Ratio} = \frac{1}{W} \sum_{k=0}^{W-1} \text{Frame Moving}_{t-k}$$
$$\text{Vehicle State} = \begin{cases} \text{DRIVING} & \text{if } \text{Moving Ratio} \ge r_{\text{min}} \\ \text{PARKED / STOPPED} & \text{otherwise} \end{cases}$$

### 1.6 Low-Light Adaptive CLAHE & Gamma Enhancement
When mean frame luminance $\bar{L} < 65.0$:
$$L_{\text{enhanced}} = \text{CLAHE}(L_{\text{channel}}, \text{clip}=2.5, \text{grid}=8\times 8)$$
$$L_{\text{gamma}} = 255 \cdot \left(\frac{L_{\text{enhanced}}}{255}\right)^\gamma \quad (\gamma = 0.6)$$

### 1.7 Eyewear Classification & Optical Reflection Models
#### A. Ocular-to-Skin Luminance Ratio ($\rho_{\text{ocular}}$)
Given mean luminance of eye socket crops $\bar{L}_{\text{eyes}}$ and upper cheek skin reference $\bar{L}_{\text{cheeks}}$:
$$\rho_{\text{ocular}} = \frac{\bar{L}_{\text{eyes}}}{\max(15.0, \bar{L}_{\text{cheeks}})}$$
$$\text{Eyewear State} = \begin{cases} 
\text{SUNGLASSES} & \text{if } \rho_{\text{ocular}} < 0.52 \land \sigma_{\text{contrast}} < 22.0 \\
\text{REGULAR\_GLASSES} & \text{if } \text{Glare Ratio} \ge 0.025 \lor (\text{Edge}_{\text{bridge}} \land \text{Glare Ratio} > 0.008) \\
\text{NONE} & \text{otherwise}
\end{cases}$$

#### B. Through-Reflection Inpainting for Clear Glasses
For prescription lenses with specular glare hotspots ($L \ge 225$):
$$M_{\text{glare}} = \text{Dilate}(\mathbb{I}(L(x, y) \ge 225), K_{3\times 3})$$
$$L_{\text{anti\_glare}} = \text{CLAHE}\left(\text{InpaintTelea}(L, M_{\text{glare}}, r=2)\right)$$

#### C. Sunglasses Surrogate Fatigue Accumulation
When $\text{Eyewear} = \text{SUNGLASSES}$, direct eye closure is bypassed:
$$\Delta \text{Score}_{\text{sunglasses}} = \alpha_{\text{mar}} \cdot \mathbb{I}(\text{MAR} > 0.48) + \beta_{\text{nod}} \cdot \mathbb{I}(\text{Pitch} < -20^\circ) + \gamma_{\text{yawn}} \cdot N_{\text{yawn\_events}} - \delta_{\text{decay}}$$
$$\text{Alert Condition} = \text{Score} \ge 75.0 \lor N_{\text{yawns}}(300\text{s}) \ge 3 \lor \text{ConsecNods} \ge 30 \text{ frames}$$

---

## 2. Database Schema (`drowsiness_tracker.db`)

### `sessions` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment session ID |
| user_label | TEXT | User identifier or profile name |
| start_time | TEXT | ISO timestamp |
| end_time | TEXT | ISO timestamp |
| baseline_ear | REAL | Recorded baseline EAR |

### `events` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment event ID |
| session_id | INTEGER FK | References `sessions(id)` |
| timestamp | TEXT | ISO timestamp |
| event_type | TEXT | `blink`, `yawn`, `head_nod`, `drowsy_warning`, `drowsy_alert`, `admin_override`, `admin_remote_override`, `admin_remote_unmute`, `user_dismiss_puzzle`, `low_battery_warning`, `camera_disconnect` |
| ear_value | REAL | EAR at event time |
| mar_value | REAL | MAR at event time |
| metadata | TEXT | Contextual info (e.g. `'PIN Authenticated'`, `'Remote Mute from 192.168.1.10'`) |

### `user_profiles` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment profile ID |
| name | TEXT UNIQUE | Profile name (e.g. 'Debasis_Normal') |
| baseline_ear | REAL | Mean open-eye EAR |
| ear_threshold | REAL | Personalized alert threshold |
| baseline_mar | REAL | Mean mouth opening |
| mar_threshold | REAL | Personalized yawn threshold |
| blink_rate | REAL | Recorded blinks/min |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

---

## 3. Configuration Schema (`config.yaml`)

```yaml
alerts:
  sound_file: assets/alarm.wav
  volume: 0.8

admin:
  pin: "1234"
  remote_enabled: true       # Enable embedded Flask web admin panel
  remote_port: 8080          # TCP port for the admin web server (Wi-Fi / LAN accessible)

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

## 4. Remote Admin & Wi-Fi REST API

| Method | Endpoint | Auth Required | Request Body | Response | Description |
|---|---|---|---|---|---|
| `GET` | `/` | No | — | HTML login page | PIN authentication form |
| `POST` | `/auth` | No | `pin=<value>` (form) | 302 redirect | Validates PIN, sets session cookie |
| `GET` | `/dashboard` | Yes | — | HTML dashboard | Live metrics, vehicle motion, buzzer controls, event log |
| `GET` | `/status` | Yes | — | JSON `{ state, fatigue_score, ear, mar, pitch, yaw, roll, is_buzzer_playing, is_low_light, is_vehicle_moving, motion_score, luminance, active_yawn_count, session_seconds, blink_rate, faces_detected }` | Current system snapshot |
| `GET` | `/log` | Yes | — | JSON `[{ event_type, timestamp }, ...]` | Last 50 events |
| `GET` | `/qr` | No | — | `image/png` stream | Dynamic QR code image encoding the Wi-Fi/LAN access URL |
| `POST` | `/buzzer/mute` | Yes | — | JSON `{ success, message, remote_ip }` | Queue alarm mute command |
| `POST` | `/buzzer/unmute` | Yes | — | JSON `{ success, message, remote_ip }` | Queue alarm re-enable command |
| `GET` | `/logout` | No | — | 302 redirect to `/` | Clear admin session |
