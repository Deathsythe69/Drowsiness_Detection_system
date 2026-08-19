"""Classifier Module.

Finite State Machine (FSM) and fatigue scoring system.
"""

from enum import Enum
import time
from collections import deque
from typing import Dict, Any, Tuple, List

class State(Enum):
    """FSM States for Drowsiness Detection."""
    AWAKE = "AWAKE"
    DROWSY_WARNING = "DROWSY_WARNING"
    DROWSY_ALERT = "DROWSY_ALERT"
    NO_FACE = "NO_FACE"

class StateClassifier:
    """FSM State Classifier and fatigue tracking logic."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize classifier with config.
        
        Args:
            config: Configuration dictionary (yaml-based).
        """
        self.update_config(config)
        self.reset()

    def update_config(self, config: Dict[str, Any]):
        """Update configurable thresholds from configuration."""
        thresholds = config.get("thresholds", {})
        self.ear_threshold = thresholds.get("ear_threshold", 0.21)
        self.ear_consec_frames = thresholds.get("ear_consec_frames", 20)
        self.mar_threshold = thresholds.get("mar_threshold", 0.6)
        self.mar_consec_frames = thresholds.get("mar_consec_frames", 15)
        self.fatigue_warning_score = thresholds.get("fatigue_warning_score", 40.0)
        self.fatigue_alert_score = thresholds.get("fatigue_alert_score", 75.0)

    def reset(self):
        """Reset state tracking buffers."""
        self.state = State.AWAKE
        self.fatigue_score = 0.0
        
        # Action tracking counters
        self.consec_eye_closed = 0
        self.consec_yawn_frames = 0
        
        # Face connection monitoring
        self.last_face_seen_time = time.time()
        
        # Historical events queue (keep 60s of logs)
        self.blink_timestamps = deque(maxlen=100)
        self.yawn_timestamps = deque(maxlen=50)
        
        # Blink parsing logic
        self.blink_in_progress = False
        self.blink_start_frame_count = 0
        self.frame_count = 0

    def process_frame(self, has_face: bool, ear: float, mar: float) -> Tuple[State, float, List[str]]:
        """Evaluate features for the frame and update FSM state.
        
        Args:
            has_face: Whether a face is detected in the current frame.
            ear: The Eye Aspect Ratio (EAR) value.
            mar: The Mouth Aspect Ratio (MAR) value.
            
        Returns:
            Tuple containing:
            - Current State (State Enum)
            - Current fatigue score (0.0 to 100.0)
            - List of event labels triggered in this frame (e.g. ['blink', 'yawn'])
        """
        self.frame_count += 1
        current_time = time.time()
        triggered_events = []

        if not has_face:
            # Check if face is lost for longer than 3 seconds
            if current_time - self.last_face_seen_time > 3.0:
                self.state = State.NO_FACE
                self.consec_eye_closed = 0
                self.consec_yawn_frames = 0
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
                # Normal human blinks are fast: 1 to 10 frames at ~30fps (~30-300ms)
                if 1 <= blink_duration <= 10:
                    self.blink_timestamps.append(current_time)
                    triggered_events.append("blink")
                self.blink_in_progress = False
            self.consec_eye_closed = 0

        # 2. Mouth Yawn Recognition
        if mar > self.mar_threshold:
            self.consec_yawn_frames += 1
        else:
            if self.consec_yawn_frames >= self.mar_consec_frames:
                self.yawn_timestamps.append(current_time)
                triggered_events.append("yawn")
                # Increment fatigue score
                self.fatigue_score = min(100.0, self.fatigue_score + 25.0)
            self.consec_yawn_frames = 0

        # 3. Dynamic Fatigue Score Processing
        # Continuous linear decay per frame
        self.fatigue_score = max(0.0, self.fatigue_score - 0.05)

        # Accumulate fatigue when eyes are shut (adds warning weight)
        if self.consec_eye_closed > 0:
            self.fatigue_score = min(100.0, self.fatigue_score + 1.5)

        # 4. Blink Rate Evaluation
        # Prune old blink timestamps (older than 60 seconds)
        while self.blink_timestamps and current_time - self.blink_timestamps[0] > 60.0:
            self.blink_timestamps.popleft()
            
        # Calculate blinks per minute
        blink_rate = len(self.blink_timestamps)
        
        # Sane threshold: normally 10-20 blinks/min. If we have at least 15s of history and blinks are very low:
        if self.frame_count > 300 and blink_rate < 5:
            self.fatigue_score = min(100.0, self.fatigue_score + 0.1)

        # 5. State Classifier Decisions
        # Rule A: Single prolonged eye closure (direct alert trigger)
        if self.consec_eye_closed >= self.ear_consec_frames:
            self.state = State.DROWSY_ALERT
            self.fatigue_score = 100.0
            triggered_events.append("drowsy_alert")
        # Rule B: Alert threshold crossed
        elif self.fatigue_score >= self.fatigue_alert_score:
            self.state = State.DROWSY_ALERT
            triggered_events.append("drowsy_alert")
        # Rule C: Warning threshold crossed, or half-closed eyes
        elif self.fatigue_score >= self.fatigue_warning_score or self.consec_eye_closed >= (self.ear_consec_frames // 2):
            # Hysteresis: don't clear alert immediately if eye is still closed
            if self.state == State.DROWSY_ALERT and self.consec_eye_closed > 0:
                self.state = State.DROWSY_ALERT
            else:
                self.state = State.DROWSY_WARNING
                triggered_events.append("drowsy_warning")
        else:
            # Exit rules with hysteresis buffer
            if self.state == State.DROWSY_ALERT and (self.consec_eye_closed > 0 or self.fatigue_score > self.fatigue_warning_score):
                self.state = State.DROWSY_ALERT
            elif self.state == State.DROWSY_WARNING and self.fatigue_score > (self.fatigue_warning_score - 10.0):
                self.state = State.DROWSY_WARNING
            else:
                self.state = State.AWAKE

        return self.state, self.fatigue_score, triggered_events
