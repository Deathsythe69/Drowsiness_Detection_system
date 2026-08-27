"""Shared State Module.

Thread-safe singleton bridge between the PyQt6 UI main thread,
the processing pipeline, and the remote Flask admin web server.

Both the UI and the Flask server read/write through this shared
state object using a threading lock for safe concurrent access.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class BuzzerCommand(Enum):
    """Commands that can be issued to the buzzer from remote admin."""
    NONE = "none"
    MUTE = "mute"
    UNMUTE = "unmute"


@dataclass
class SystemMetrics:
    """Snapshot of current system metrics for remote dashboard display."""
    state: str = "INITIALIZING"
    fatigue_score: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    is_buzzer_playing: bool = False
    is_low_light: bool = False
    luminance: float = 100.0
    active_yawn_count: int = 0
    session_seconds: int = 0
    blink_rate: int = 0
    faces_detected: int = 0
    is_vehicle_moving: bool = False
    motion_score: float = 0.0
    is_simulated_driving: bool = False
    perclos: float = 0.0
    latest_evidence_file: str = ""
    eyewear_type: str = "NONE"
    shoulder_angle: float = 0.0
    is_slouched: bool = False
    is_frozen_still: bool = False
    pose_staleness_frames: int = 0


class SharedState:
    """Thread-safe shared state singleton for cross-thread communication.
    
    Provides safe read/write access to system metrics and buzzer
    commands between the PyQt6 UI thread and the Flask admin server thread.
    """

    _instance: Optional["SharedState"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "SharedState":
        """Ensure singleton pattern."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize shared state (only once due to singleton)."""
        if self._initialized:
            return
        self._initialized = True
        self._lock = threading.Lock()
        self._metrics = SystemMetrics()
        self._pending_command = BuzzerCommand.NONE
        self._remote_override_log: List[Dict[str, Any]] = []
        self._recent_events: List[Dict[str, str]] = []
        self._remote_admin_url: str = ""
        self._is_monitoring: bool = False
        self._latest_evidence_path: str = ""

    def set_latest_evidence(self, filepath: str):
        """Register the latest recorded evidence video file."""
        with self._lock:
            self._latest_evidence_path = filepath
            self._metrics.latest_evidence_file = os.path.basename(filepath)

    def get_latest_evidence(self) -> str:
        """Get the absolute/relative path of the most recent evidence video."""
        with self._lock:
            return self._latest_evidence_path

    def update_metrics(
        self,
        state: str,
        fatigue_score: float,
        ear: float,
        mar: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        is_buzzer_playing: bool = False,
        is_low_light: bool = False,
        luminance: float = 100.0,
        active_yawn_count: int = 0,
        session_seconds: int = 0,
        blink_rate: int = 0,
        faces_detected: int = 0,
        is_vehicle_moving: bool = False,
        motion_score: float = 0.0,
        is_simulated_driving: bool = False,
        perclos: float = 0.0,
        latest_evidence_file: Optional[str] = None,
        eyewear_type: str = "NONE",
        shoulder_angle: float = 0.0,
        is_slouched: bool = False,
        is_frozen_still: bool = False,
        pose_staleness_frames: int = 0
    ):
        """Update current system metrics (called from UI/processing thread).
        
        Args:
            state: Current FSM state string (e.g., 'AWAKE', 'DROWSY_ALERT').
            fatigue_score: Current fatigue accumulation (0-100).
            ear: Current Eye Aspect Ratio.
            mar: Current Mouth Aspect Ratio.
            pitch: Head pitch angle in degrees.
            yaw: Head yaw angle in degrees.
            roll: Head roll angle in degrees.
            is_buzzer_playing: Whether the alarm is currently sounding.
            is_low_light: Whether low-light mode is active.
            luminance: Current frame luminance value.
            active_yawn_count: Rolling yawn count in current window.
            session_seconds: Current session duration in seconds.
            blink_rate: Blinks per minute over last 60s.
            faces_detected: Number of faces currently detected.
            is_vehicle_moving: Whether vehicle is detected as moving.
            motion_score: Peripheral optical flow motion score.
            is_simulated_driving: Whether driving mode is simulated for desk testing.
            perclos: Percentage of eye closure ratio (0.0 to 1.0).
            latest_evidence_file: Filename of the newest evidence video clip.
            eyewear_type: Driver eyewear classification (NONE, REGULAR_GLASSES, SUNGLASSES).
            shoulder_angle: Upper body shoulder-line tilt angle in degrees.
            is_slouched: Whether driver posture is currently slouched.
            is_frozen_still: Whether posture indicates rigid stillness.
            pose_staleness_frames: Number of frames since last real Pose inference.
        """
        with self._lock:
            self._metrics.state = state
            self._metrics.fatigue_score = fatigue_score
            self._metrics.ear = ear
            self._metrics.mar = mar
            self._metrics.pitch = pitch
            self._metrics.yaw = yaw
            self._metrics.roll = roll
            self._metrics.is_buzzer_playing = is_buzzer_playing
            self._metrics.is_low_light = is_low_light
            self._metrics.luminance = luminance
            self._metrics.active_yawn_count = active_yawn_count
            self._metrics.session_seconds = session_seconds
            self._metrics.blink_rate = blink_rate
            self._metrics.faces_detected = faces_detected
            self._metrics.is_vehicle_moving = is_vehicle_moving
            self._metrics.motion_score = motion_score
            self._metrics.is_simulated_driving = is_simulated_driving
            self._metrics.perclos = perclos
            self._metrics.eyewear_type = eyewear_type
            self._metrics.shoulder_angle = shoulder_angle
            self._metrics.is_slouched = is_slouched
            self._metrics.is_frozen_still = is_frozen_still
            self._metrics.pose_staleness_frames = pose_staleness_frames
            if latest_evidence_file:
                self._metrics.latest_evidence_file = latest_evidence_file

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics as a dictionary (called from Flask thread).
        
        Returns:
            Dict with all current system metric values.
        """
        with self._lock:
            m = self._metrics
            return {
                "state": m.state,
                "fatigue_score": round(m.fatigue_score, 1),
                "ear": round(m.ear, 3),
                "mar": round(m.mar, 3),
                "pitch": round(m.pitch, 1),
                "yaw": round(m.yaw, 1),
                "roll": round(m.roll, 1),
                "is_buzzer_playing": m.is_buzzer_playing,
                "is_low_light": m.is_low_light,
                "luminance": round(m.luminance, 1),
                "active_yawn_count": m.active_yawn_count,
                "session_seconds": m.session_seconds,
                "blink_rate": m.blink_rate,
                "faces_detected": m.faces_detected,
                "is_vehicle_moving": m.is_vehicle_moving,
                "motion_score": round(m.motion_score, 2),
                "is_simulated_driving": m.is_simulated_driving,
                "perclos": round(m.perclos, 3),
                "latest_evidence_file": m.latest_evidence_file,
                "eyewear_type": m.eyewear_type,
                "shoulder_angle": round(m.shoulder_angle, 1),
                "is_slouched": m.is_slouched,
                "is_frozen_still": m.is_frozen_still,
                "pose_staleness_frames": m.pose_staleness_frames
            }

    def set_buzzer_command(self, command: BuzzerCommand, remote_ip: str = ""):
        """Queue a buzzer command from the remote admin panel.
        
        Args:
            command: The buzzer action to perform.
            remote_ip: IP address of the remote admin client.
        """
        with self._lock:
            self._pending_command = command
            if command != BuzzerCommand.NONE and remote_ip:
                self._remote_override_log.append({
                    "command": command.value,
                    "remote_ip": remote_ip
                })

    def consume_buzzer_command(self) -> tuple:
        """Consume and clear any pending buzzer command (called from UI thread).
        
        Returns:
            Tuple of (BuzzerCommand, remote_ip or empty string).
        """
        with self._lock:
            cmd = self._pending_command
            remote_ip = ""
            if self._remote_override_log:
                remote_ip = self._remote_override_log[-1].get("remote_ip", "")
            self._pending_command = BuzzerCommand.NONE
            return cmd, remote_ip

    def push_event(self, event_type: str, timestamp: str):
        """Push an event to the recent events buffer for remote display.
        
        Args:
            event_type: Type of event (e.g., 'yawn', 'blink', 'drowsy_alert').
            timestamp: ISO timestamp string.
        """
        with self._lock:
            self._recent_events.append({
                "event_type": event_type,
                "timestamp": timestamp
            })
            # Keep only last 50 events
            if len(self._recent_events) > 50:
                self._recent_events = self._recent_events[-50:]

    def get_recent_events(self) -> List[Dict[str, str]]:
        """Get recent events for the remote dashboard.
        
        Returns:
            List of recent event dictionaries.
        """
        with self._lock:
            return list(self._recent_events)

    def set_remote_admin_url(self, url: str):
        """Store the remote admin panel URL for display in UI.
        
        Args:
            url: The full URL (e.g., 'http://192.168.1.5:8080').
        """
        with self._lock:
            self._remote_admin_url = url

    def get_remote_admin_url(self) -> str:
        """Get the remote admin URL.
        
        Returns:
            The remote admin URL string.
        """
        with self._lock:
            return self._remote_admin_url

    def set_monitoring(self, active: bool):
        """Set monitoring state.
        
        Args:
            active: Whether monitoring is currently active.
        """
        with self._lock:
            self._is_monitoring = active

    def is_monitoring(self) -> bool:
        """Check if monitoring is active.
        
        Returns:
            True if monitoring is currently running.
        """
        with self._lock:
            return self._is_monitoring

    @classmethod
    def reset_singleton(cls):
        """Reset the singleton instance (useful for testing).
        
        WARNING: Only use in test environments.
        """
        with cls._init_lock:
            cls._instance = None
