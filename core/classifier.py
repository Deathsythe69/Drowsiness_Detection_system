"""Classifier Module with Rolling Yawn Frequency and Posture Integration.

Finite State Machine (FSM) evaluating multi-modal physiological features:
- Eye closure duration & personalized blink frequency
- Rolling time-window yawn frequency escalation
- Head nod / droop and posture distraction penalties
"""

from enum import Enum
import time
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

from core.eyewear_detector import EyewearType

class State(Enum):
    """FSM States for Drowsiness Detection."""
    AWAKE = "AWAKE"
    DROWSY_WARNING = "DROWSY_WARNING"
    DROWSY_ALERT = "DROWSY_ALERT"
    NO_FACE = "NO_FACE"

class StateClassifier:
    """Multi-signal FSM Classifier and fatigue accumulator."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize classifier with config.
        
        Args:
            config: Configuration dictionary (yaml-based).
        """
        self.config = config
        self.current_eyewear_type = EyewearType.NONE
        self.update_config(config)
        self.reset()

    def update_config(self, config: Dict[str, Any]):
        """Update configurable thresholds from configuration."""
        self.config = config
        thresholds = config.get("thresholds", {})
        self.ear_threshold = thresholds.get("ear_threshold", 0.16)
        self.ear_consec_frames = thresholds.get("ear_consec_frames", 25)
        self.max_blink_frames = thresholds.get("max_blink_frames", 15)
        self.mar_threshold = thresholds.get("mar_threshold", 0.65)
        self.mar_consec_frames = thresholds.get("mar_consec_frames", 20)
        self.fatigue_warning_score = thresholds.get("fatigue_warning_score", 40.0)
        self.fatigue_alert_score = thresholds.get("fatigue_alert_score", 75.0)
        self.perclos_threshold = thresholds.get("perclos_threshold", 0.35)
        
        # Yawn frequency tracking parameters
        self.yawn_window_seconds = thresholds.get("yawn_window_seconds", 300)
        self.yawn_frequency_alert_threshold = thresholds.get("yawn_frequency_alert_threshold", 3)
        
        # Posture parameters
        posture_cfg = config.get("posture", {})
        self.posture_enabled = posture_cfg.get("enabled", True)
        self.pitch_nod_threshold = posture_cfg.get("pitch_nod_threshold", -22.0)
        self.yaw_distraction_threshold = posture_cfg.get("yaw_distraction_threshold", 30.0)
        self.slouch_penalty_weight = posture_cfg.get("slouch_penalty_weight", 0.20)
        
        # Eyewear & Glare parameters
        eyewear_cfg = config.get("eyewear", {})
        self.glare_max_blind_frames = eyewear_cfg.get("glare_max_blind_frames", 60)
        self.consec_glare_frames = 0
        self.is_eye_state_unknown = False

    def reset(self):
        """Reset all state tracking buffers."""
        self.state = State.AWAKE
        self.fatigue_score = 0.0
        self.current_eyewear_type = EyewearType.NONE
        
        # Action tracking counters
        self.consec_eye_closed = 0
        self.consec_yawn_frames = 0
        self.consec_nod_frames = 0
        self.consec_glare_frames = 0
        self.is_eye_state_unknown = False
        
        # Face connection monitoring
        self.last_face_seen_time = time.time()
        
        # Rolling event history queues
        self.blink_timestamps = deque(maxlen=150)
        self.yawn_timestamps = deque(maxlen=100)
        self.perclos_window_60 = deque(maxlen=60)
        self.perclos_window_300 = deque(maxlen=300)
        
        # Blink parsing logic
        self.blink_in_progress = False
        self.blink_start_frame_count = 0
        self.frame_count = 0
        
        # Latest metrics
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.active_yawn_count = 0
        self.perclos_60 = 0.0
        self.perclos_300 = 0.0
        
        # Exponential Moving Average (EMA) smoothing for feature stability
        self.ema_alpha = 0.65
        self.ema_ear: Optional[float] = None
        self.ema_mar: Optional[float] = None
        self.ema_pitch: Optional[float] = None
        self.ema_yaw: Optional[float] = None

    def process_frame(
        self,
        has_face: bool,
        ear: float,
        mar: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        eye_open_prob: Optional[float] = None,
        current_time: Optional[float] = None,
        eyewear_type: EyewearType = EyewearType.NONE,
        eye_state_unknown: bool = False,
        is_glare_occluded: bool = False
    ) -> Tuple[State, float, List[str]]:
        """Evaluate features for the frame and update FSM state.
        
        Args:
            has_face: Whether primary face is detected.
            ear: Eye Aspect Ratio.
            mar: Mouth Aspect Ratio.
            pitch: Head pitch angle (degrees).
            yaw: Head yaw angle (degrees).
            roll: Head roll angle (degrees).
            eye_open_prob: Optional deep neural eye openness probability (0.0 to 1.0).
            eyewear_type: Detected driver eyewear state (NONE, REGULAR_GLASSES, SUNGLASSES).
            
        Returns:
            Tuple containing:
            - Current State (State Enum)
            - Current fatigue score (0.0 to 100.0)
            - List of event labels triggered in this frame
        """
        self.frame_count += 1
        self.current_eyewear_type = eyewear_type
        current_time = time.time()
        triggered_events = []

        if not has_face:
            # Check if face is lost for longer than 3 seconds
            if current_time - self.last_face_seen_time > 3.0:
                self.state = State.NO_FACE
                self.consec_eye_closed = 0
                self.consec_yawn_frames = 0
                self.consec_nod_frames = 0
                return self.state, self.fatigue_score, triggered_events
            return self.state, self.fatigue_score, triggered_events

        # Face is present, update timestamp
        self.last_face_seen_time = current_time

        # 1. Feature Smoothing (EMA) to suppress single-frame sensor jitter
        if self.ema_ear is None:
            self.ema_ear = ear
            self.ema_mar = mar
            self.ema_pitch = pitch
            self.ema_yaw = yaw
        else:
            self.ema_ear = (self.ema_alpha * ear) + ((1.0 - self.ema_alpha) * self.ema_ear)
            self.ema_mar = (self.ema_alpha * mar) + ((1.0 - self.ema_alpha) * self.ema_mar)
            self.ema_pitch = (self.ema_alpha * pitch) + ((1.0 - self.ema_alpha) * self.ema_pitch)
            self.ema_yaw = (self.ema_alpha * yaw) + ((1.0 - self.ema_alpha) * self.ema_yaw)

        smooth_ear = float(self.ema_ear)
        smooth_mar = float(self.ema_mar)
        smooth_pitch = float(self.ema_pitch)
        smooth_yaw = float(self.ema_yaw)

        self.current_pitch = smooth_pitch
        self.current_yaw = smooth_yaw
        self.current_roll = roll

        # Branch between Sunglasses Surrogate Fatigue Mode vs Direct Eye Tracking Mode
        if eyewear_type == EyewearType.SUNGLASSES:
            # Dark sunglasses occlude direct eye visibility.
            # Bypass eye closure / PERCLOS to prevent dark-lens false alarms!
            self.consec_eye_closed = 0
            self.perclos_60 = 0.0
            self.perclos_300 = 0.0
            
            # --- SUNGLASSES SURROGATE MODE (MAR + PITCH + YAW) ---
            # 1. Mouth Opening (MAR) & Yawn Kinetics
            if smooth_mar > self.mar_threshold:
                self.consec_yawn_frames += 1
            else:
                if self.consec_yawn_frames >= self.mar_consec_frames:
                    self.yawn_timestamps.append(current_time)
                    triggered_events.append("yawn")
                    self.fatigue_score = min(100.0, self.fatigue_score + 35.0)
                self.consec_yawn_frames = 0

            # Prune old yawns
            while self.yawn_timestamps and (current_time - self.yawn_timestamps[0]) > self.yawn_window_seconds:
                self.yawn_timestamps.popleft()
            self.active_yawn_count = len(self.yawn_timestamps)

            # Slack jaw / sustained mouth breathing in sunglasses
            if smooth_mar > 0.48:
                self.fatigue_score = min(100.0, self.fatigue_score + 0.60)
                if "sunglasses_slack_jaw" not in triggered_events:
                    triggered_events.append("sunglasses_slack_jaw")

            # 2. solvePnP 3D Head Pose (Pitch nodding & Yaw drift)
            if self.posture_enabled:
                # Downward nodding (nodding off while driving with sunglasses)
                if smooth_pitch < self.pitch_nod_threshold:
                    self.consec_nod_frames += 1
                    if self.consec_nod_frames >= 12:
                        # Accumulate nod penalty up to alert level
                        self.fatigue_score = min(100.0, self.fatigue_score + 0.85)
                        if "head_nod" not in triggered_events:
                            triggered_events.append("head_nod")
                        if self.consec_nod_frames >= 30 or smooth_pitch < (self.pitch_nod_threshold - 6.0):
                            self.fatigue_score = max(self.fatigue_score, 80.0)
                            if "severe_head_nod" not in triggered_events:
                                triggered_events.append("severe_head_nod")
                else:
                    self.consec_nod_frames = 0

                # Distraction & lateral head wobble (Yaw)
                if abs(smooth_yaw) > self.yaw_distraction_threshold:
                    self.fatigue_score = min(100.0, self.fatigue_score + 0.40)
                    if "yaw_distraction" not in triggered_events:
                        triggered_events.append("yaw_distraction")

            # 3. Compound Drowsiness in Sunglasses (Yawn / Open Mouth + Head Nod)
            if (smooth_mar > 0.45 or self.consec_yawn_frames > 5) and smooth_pitch < (self.pitch_nod_threshold + 3.0):
                self.fatigue_score = min(100.0, self.fatigue_score + 1.20)
                if "compound_drowsiness" not in triggered_events:
                    triggered_events.append("compound_drowsiness")

            # 4. Natural Fatigue Decay (Driver alert & upright)
            if smooth_mar <= 0.35 and smooth_pitch >= (self.pitch_nod_threshold + 5.0) and abs(smooth_yaw) <= self.yaw_distraction_threshold:
                self.fatigue_score = max(0.0, self.fatigue_score - 0.25)

            # 5. Sunglasses FSM State Decision
            if self.fatigue_score >= self.fatigue_alert_score or self.active_yawn_count >= 3 or self.consec_nod_frames >= 30:
                self.state = State.DROWSY_ALERT
                triggered_events.append("drowsy_alert")
                triggered_events.append("sunglasses_mode")
            elif self.fatigue_score >= self.fatigue_warning_score or self.active_yawn_count >= 2 or self.consec_nod_frames >= 15 or self.consec_yawn_frames >= 10:
                self.state = State.DROWSY_WARNING
                triggered_events.append("drowsy_warning")
                triggered_events.append("sunglasses_mode")
            else:
                self.state = State.AWAKE

            return self.state, self.fatigue_score, triggered_events

        # --- DIRECT EYE TRACKING MODE (NONE / REGULAR_GLASSES) ---
        self.is_eye_state_unknown = eye_state_unknown or is_glare_occluded
        is_eye_clearly_open = False

        if self.is_eye_state_unknown:
            # Eye region is glare-saturated / occluded.
            # Treat frames as "eye state unknown": do NOT feed corrupted EAR into PERCLOS or treat as open!
            self.consec_glare_frames += 1
            if "eye_region_glare" not in triggered_events:
                triggered_events.append("eye_region_glare")
            
            # Cancel active blink without logging a fake blink
            self.blink_in_progress = False
            
            # If glare blindness persists across a prolonged run (e.g. >= 60 frames / ~2s),
            # cautiously escalate safety score up to warning level so we never silently fail
            if self.consec_glare_frames >= self.glare_max_blind_frames:
                self.fatigue_score = min(self.fatigue_warning_score, self.fatigue_score + 0.25)
                if "eye_glare_blindness_warning" not in triggered_events:
                    triggered_events.append("eye_glare_blindness_warning")
        else:
            self.consec_glare_frames = 0
            # 2. Eye Closure & Blink Recognition (Dual-Signal: EAR + Neural Probability)
            is_ear_closed = smooth_ear < self.ear_threshold
            if eye_open_prob is not None:
                # Neural model gate:
                # Override false low EAR when neural model is confident eyes are open (>= 0.70)
                # Or flag closed if neural model detects closure (< 0.35)
                is_eye_closed = (is_ear_closed and eye_open_prob < 0.70) or (eye_open_prob < 0.35)
                is_eye_clearly_open = eye_open_prob >= 0.65 and smooth_ear >= self.ear_threshold
            else:
                is_eye_closed = is_ear_closed
                is_eye_clearly_open = smooth_ear >= self.ear_threshold

            if is_eye_closed:
                self.consec_eye_closed += 1
                self.perclos_window_60.append(1.0)
                self.perclos_window_300.append(1.0)
                if not self.blink_in_progress:
                    self.blink_in_progress = True
                    self.blink_start_frame_count = self.frame_count
            else:
                self.perclos_window_60.append(0.0)
                self.perclos_window_300.append(0.0)
                if self.blink_in_progress:
                    self.blink_in_progress = False
                    blink_duration = self.frame_count - self.blink_start_frame_count
                    max_bf = getattr(self, "max_blink_frames", 15)
                    if 1 <= blink_duration <= max_bf:
                        self.blink_timestamps.append(current_time)
                        triggered_events.append("blink")
                self.consec_eye_closed = 0

        # Calculate Automotive-Standard PERCLOS (Percentage of Eye Closure)
        if len(self.perclos_window_60) > 0:
            self.perclos_60 = float(sum(self.perclos_window_60) / len(self.perclos_window_60))
        if len(self.perclos_window_300) > 0:
            self.perclos_300 = float(sum(self.perclos_window_300) / len(self.perclos_window_300))

        # 3. Mouth Yawn Recognition & Rolling Window Frequency
        if smooth_mar > self.mar_threshold:
            self.consec_yawn_frames += 1
        else:
            if self.consec_yawn_frames >= self.mar_consec_frames:
                self.yawn_timestamps.append(current_time)
                triggered_events.append("yawn")
                # Base yawn score increment
                self.fatigue_score = min(100.0, self.fatigue_score + 25.0)
            self.consec_yawn_frames = 0

        # Prune yawn timestamps older than rolling window
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0]) > self.yawn_window_seconds:
            self.yawn_timestamps.popleft()
        self.active_yawn_count = len(self.yawn_timestamps)

        # Escalate fatigue if yawn frequency crosses threshold in rolling window
        if self.active_yawn_count >= self.yawn_frequency_alert_threshold:
            self.fatigue_score = min(100.0, self.fatigue_score + 10.0)
            triggered_events.append("yawn_frequency_escalation")

        # 4. Harmonized Sweet-Spot Posture & Distraction Multi-Ratio Fusion
        # Cross-Feature Gating: If eyes are clearly open, looking down/around is attenuated by 85%
        attenuation = 0.15 if is_eye_clearly_open else 1.0

        if self.posture_enabled:
            # Check for downward head nod (Pitch negative)
            if smooth_pitch < self.pitch_nod_threshold:
                self.consec_nod_frames += 1
                if self.consec_nod_frames >= 20:
                    # Cap posture penalty to max 35.0 so looking down alone cannot ring the buzzer
                    nod_penalty = (self.slouch_penalty_weight * 0.20) * attenuation
                    self.fatigue_score = min(35.0, self.fatigue_score + nod_penalty)
                    if "head_nod" not in triggered_events:
                        triggered_events.append("head_nod")
            else:
                self.consec_nod_frames = 0

            # Distraction penalty (Yaw turned away) - capped to warning level
            if abs(smooth_yaw) > self.yaw_distraction_threshold:
                distraction_penalty = (self.slouch_penalty_weight * 0.10) * attenuation
                self.fatigue_score = min(30.0, self.fatigue_score + distraction_penalty)

        # 5. Compound Drowsiness Synergy (Drooping Eyes + Head Nod)
        is_drooping = (self.ear_threshold - 0.03) <= smooth_ear <= (self.ear_threshold + 0.02)
        is_nodding = smooth_pitch < (self.pitch_nod_threshold + 4.0)
        if is_drooping and is_nodding and not is_eye_clearly_open:
            self.fatigue_score = min(100.0, self.fatigue_score + 0.50)
            if "compound_drowsiness" not in triggered_events:
                triggered_events.append("compound_drowsiness")

        # 6. Continuous Decay & Eye Closure Acceleration + PERCLOS Fatigue Penalty
        if self.consec_eye_closed > 0:
            # Eyes closed: accumulate fatigue
            self.fatigue_score = min(100.0, self.fatigue_score + 1.20)
        elif not self.is_eye_state_unknown:
            # Eyes confirmed open: decay fatigue back toward 0.0
            decay_rate = 0.20 if self.perclos_60 > 0.25 else 0.40
            self.fatigue_score = max(0.0, self.fatigue_score - decay_rate)

        # PERCLOS sustained fatigue injection (early drowsiness warning before full microsleep)
        if len(self.perclos_window_60) >= 30 and self.perclos_60 >= self.perclos_threshold:
            self.fatigue_score = min(100.0, self.fatigue_score + 0.80)
            if "perclos_fatigue" not in triggered_events:
                triggered_events.append("perclos_fatigue")

        # 7. Blink Rate Evaluation
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60.0:
            self.blink_timestamps.popleft()

        # 8. FSM State Decision
        # Condition A: Sustained closed eyes (Microsleep / Asleep)
        if self.consec_eye_closed >= self.ear_consec_frames:
            self.state = State.DROWSY_ALERT
            self.fatigue_score = 100.0
            triggered_events.append("drowsy_alert")
        # Condition B: High fatigue score (>= fatigue_alert_score)
        elif self.fatigue_score >= self.fatigue_alert_score:
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Condition C: Severe yawn frequency alert
        elif self.active_yawn_count >= (self.yawn_frequency_alert_threshold + 2):
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Condition D: High PERCLOS Alert (Prolonged drowsiness)
        elif len(self.perclos_window_60) >= 45 and self.perclos_60 >= (self.perclos_threshold + 0.15):
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Condition E: Warning State (Half-closed eyes, mild fatigue, elevated PERCLOS, or moderate yawning)
        elif (self.fatigue_score >= self.fatigue_warning_score or 
              self.consec_eye_closed >= (self.ear_consec_frames // 2) or
              (len(self.perclos_window_60) >= 30 and self.perclos_60 >= self.perclos_threshold) or
              self.active_yawn_count >= self.yawn_frequency_alert_threshold):
            self.state = State.DROWSY_WARNING
            triggered_events.append("drowsy_warning")
        else:
            self.state = State.AWAKE

        return self.state, self.fatigue_score, triggered_events
