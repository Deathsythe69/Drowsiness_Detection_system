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
        self.update_config(config)
        self.reset()

    def update_config(self, config: Dict[str, Any]):
        """Update configurable thresholds from configuration."""
        self.config = config
        thresholds = config.get("thresholds", {})
        self.ear_threshold = thresholds.get("ear_threshold", 0.21)
        self.ear_consec_frames = thresholds.get("ear_consec_frames", 20)
        self.mar_threshold = thresholds.get("mar_threshold", 0.60)
        self.mar_consec_frames = thresholds.get("mar_consec_frames", 15)
        self.fatigue_warning_score = thresholds.get("fatigue_warning_score", 40.0)
        self.fatigue_alert_score = thresholds.get("fatigue_alert_score", 75.0)
        
        # Yawn frequency tracking parameters
        self.yawn_window_seconds = thresholds.get("yawn_window_seconds", 300)
        self.yawn_frequency_alert_threshold = thresholds.get("yawn_frequency_alert_threshold", 3)
        
        # Posture parameters
        posture_cfg = config.get("posture", {})
        self.posture_enabled = posture_cfg.get("enabled", True)
        self.pitch_nod_threshold = posture_cfg.get("pitch_nod_threshold", -18.0)
        self.yaw_distraction_threshold = posture_cfg.get("yaw_distraction_threshold", 25.0)
        self.slouch_penalty_weight = posture_cfg.get("slouch_penalty_weight", 1.0)

    def reset(self):
        """Reset all state tracking buffers."""
        self.state = State.AWAKE
        self.fatigue_score = 0.0
        
        # Action tracking counters
        self.consec_eye_closed = 0
        self.consec_yawn_frames = 0
        self.consec_nod_frames = 0
        
        # Face connection monitoring
        self.last_face_seen_time = time.time()
        
        # Rolling event history queues
        self.blink_timestamps = deque(maxlen=150)
        self.yawn_timestamps = deque(maxlen=100)
        
        # Blink parsing logic
        self.blink_in_progress = False
        self.blink_start_frame_count = 0
        self.frame_count = 0
        
        # Latest metrics
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.active_yawn_count = 0

    def process_frame(
        self,
        has_face: bool,
        ear: float,
        mar: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0
    ) -> Tuple[State, float, List[str]]:
        """Evaluate features for the frame and update FSM state.
        
        Args:
            has_face: Whether primary face is detected.
            ear: Eye Aspect Ratio.
            mar: Mouth Aspect Ratio.
            pitch: Head pitch angle (degrees).
            yaw: Head yaw angle (degrees).
            roll: Head roll angle (degrees).
            
        Returns:
            Tuple containing:
            - Current State (State Enum)
            - Current fatigue score (0.0 to 100.0)
            - List of event labels triggered in this frame
        """
        self.frame_count += 1
        current_time = time.time()
        triggered_events = []
        self.current_pitch = pitch
        self.current_yaw = yaw
        self.current_roll = roll

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

        # 1. Eye Closure & Blink Recognition
        if ear < self.ear_threshold:
            self.consec_eye_closed += 1
            if not self.blink_in_progress:
                self.blink_in_progress = True
                self.blink_start_frame_count = self.frame_count
        else:
            if self.blink_in_progress:
                blink_duration = self.frame_count - self.blink_start_frame_count
                # Normal blinks: 1 to 10 frames (~30-300ms)
                if 1 <= blink_duration <= 10:
                    self.blink_timestamps.append(current_time)
                    triggered_events.append("blink")
                self.blink_in_progress = False
            self.consec_eye_closed = 0

        # 2. Mouth Yawn Recognition & Rolling Window Frequency
        if mar > self.mar_threshold:
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
            self.fatigue_score = min(100.0, self.fatigue_score + 15.0)
            triggered_events.append("yawn_frequency_escalation")

        # 3. Posture / Head Pose Evaluation (Secondary Signal)
        if self.posture_enabled:
            # Check for downward head nod (Pitch severely negative)
            if pitch < self.pitch_nod_threshold:
                self.consec_nod_frames += 1
                if self.consec_nod_frames >= 10:
                    self.fatigue_score = min(100.0, self.fatigue_score + self.slouch_penalty_weight)
                    if "head_nod" not in triggered_events:
                        triggered_events.append("head_nod")
            else:
                self.consec_nod_frames = 0

            # Distraction penalty (Yaw turned away)
            if abs(yaw) > self.yaw_distraction_threshold:
                self.fatigue_score = min(100.0, self.fatigue_score + (self.slouch_penalty_weight * 0.5))

        # 4. Continuous Decay & Eye Closure Acceleration
        self.fatigue_score = max(0.0, self.fatigue_score - 0.05)
        if self.consec_eye_closed > 0:
            self.fatigue_score = min(100.0, self.fatigue_score + 1.5)

        # 5. Blink Rate Evaluation (Blik/min over last 60s)
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60.0:
            self.blink_timestamps.popleft()
            
        blink_rate = len(self.blink_timestamps)
        if self.frame_count > 300 and blink_rate < 5:
            self.fatigue_score = min(100.0, self.fatigue_score + 0.1)

        # 6. FSM State Decision
        # Condition A: Single prolonged eye closure
        if self.consec_eye_closed >= self.ear_consec_frames:
            self.state = State.DROWSY_ALERT
            self.fatigue_score = 100.0
            triggered_events.append("drowsy_alert")
        # Condition B: High yawn frequency escalation threshold
        elif self.active_yawn_count >= (self.yawn_frequency_alert_threshold + 1):
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Condition C: Fatigue alert score crossed
        elif self.fatigue_score >= self.fatigue_alert_score:
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Condition D: Warning score or half-closed eyes or moderate yawn frequency
        elif (self.fatigue_score >= self.fatigue_warning_score or 
              self.consec_eye_closed >= (self.ear_consec_frames // 2) or
              self.active_yawn_count >= self.yawn_frequency_alert_threshold):
            if self.state == State.DROWSY_ALERT and self.consec_eye_closed > 0:
                self.state = State.DROWSY_ALERT
            else:
                self.state = State.DROWSY_WARNING
                triggered_events.append("drowsy_warning")
        else:
            # Hysteresis recovery
            if self.state == State.DROWSY_ALERT and (self.consec_eye_closed > 0 or self.fatigue_score > self.fatigue_warning_score):
                self.state = State.DROWSY_ALERT
            elif self.state == State.DROWSY_WARNING and self.fatigue_score > (self.fatigue_warning_score - 10.0):
                self.state = State.DROWSY_WARNING
            else:
                self.state = State.AWAKE

        return self.state, self.fatigue_score, triggered_events
