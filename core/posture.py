"""Full-Body Posture Analysis Module (MediaPipe Pose).

Extracts upper-body and torso landmarks using MediaPipe Pose to compute:
1. Shoulder-line tilt angle (deviation from horizontal slouch/lean).
2. Head-to-shoulder droop delta (combining head orientation with torso posture).
3. Rolling movement variance (detecting unnatural stillness / frozen rigid posture).
4. Multi-person driver ROI filtering and adaptive frame-skipping / throttling.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PostureMetrics:
    """Container for upper-body and posture metrics."""
    shoulder_angle: float = 0.0          # Shoulder-line tilt in degrees from horizontal
    is_slouched: bool = False             # Whether shoulder tilt exceeds slouch threshold
    head_shoulder_distance: float = 0.0  # Vertical distance between head center and mid-shoulder
    is_drooping_combined: bool = False   # Head pitch droop combined with torso compression
    stillness_variance: float = 0.0      # Rolling movement variance of upper body
    is_frozen_still: bool = False        # Whether stillness variance indicates rigid posture
    posture_penalty: float = 0.0         # Derived secondary fatigue penalty [0.0, 100.0]
    staleness_frames: int = 0            # Number of frames since last real Pose inference
    is_driver_pose: bool = True          # Whether pose belongs to primary driver ROI
    landmarks: Optional[np.ndarray] = None # (33, 3) normalized pose coordinates


class PostureDetector:
    """MediaPipe Pose wrapper for real-time driver slouch, droop, and stillness tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MediaPipe Pose and posture tracking queues.
        
        Args:
            config: Loaded system configuration dictionary.
        """
        self.config = config or {}
        posture_cfg = self.config.get("posture", {})
        
        self.enabled = posture_cfg.get("enabled", True)
        self.slouch_angle_threshold = float(posture_cfg.get("slouch_angle_threshold", 12.0))
        self.head_shoulder_droop_threshold = float(posture_cfg.get("head_shoulder_droop_threshold", 0.18))
        self.stillness_window_seconds = int(posture_cfg.get("stillness_window_seconds", 10))
        self.stillness_variance_threshold = float(posture_cfg.get("stillness_variance_threshold", 0.00015))
        self.posture_score_weight = float(posture_cfg.get("posture_score_weight", 1.0))
        self.model_complexity = int(posture_cfg.get("pose_model_complexity", 0))
        self.inference_interval = int(posture_cfg.get("pose_inference_interval_frames", 3))

        # MediaPipe Pose instance (lightweight model_complexity=0 for CPU efficiency)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Rolling movement history queue for stillness / fidget variance
        # Default ~30 FPS * 10 seconds = 300 samples
        self.history_max_len = max(30, self.stillness_window_seconds * 30)
        self.shoulder_pos_history = deque(maxlen=self.history_max_len)

        # Frame throttling cache
        self.frame_count: int = 0
        self.cached_metrics = PostureMetrics()

    def process_frame(
        self,
        frame: np.ndarray,
        pitch: float = 0.0,
        driver_face_bbox: Optional[Tuple[int, int, int, int]] = None,
        driver_roi_config: Optional[Dict[str, float]] = None,
        force_inference: bool = False
    ) -> PostureMetrics:
        """Run upper-body posture analysis with multi-person gating and throttling.
        
        Args:
            frame: OpenCV BGR video frame.
            pitch: Head pitch angle in degrees from solvePnP (negative = looking down).
            driver_face_bbox: Optional (x, y, w, h) in pixels of primary driver face.
            driver_roi_config: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max' normalized bounds.
            force_inference: If True, bypass frame-skipping interval.
            
        Returns:
            PostureMetrics instance with calculated slouch, droop, and stillness values.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return PostureMetrics()

        self.frame_count += 1
        h, w = frame.shape[:2]

        # Throttling check: Run full MediaPipe Pose every N frames, holding/interpolating between
        should_run = force_inference or (self.frame_count % self.inference_interval == 0)
        if not should_run and self.cached_metrics.landmarks is not None:
            # Increment staleness and return cached posture metrics
            self.cached_metrics.staleness_frames += 1
            return self.cached_metrics

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if not results.pose_landmarks:
                self.cached_metrics = PostureMetrics(staleness_frames=self.cached_metrics.staleness_frames + 1)
                return self.cached_metrics

            # Extract normalized 33 pose landmarks (N, 3)
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)

            # 1. Multi-Person Driver ROI Validation
            # Pose landmarks: 11 = LEFT_SHOULDER, 12 = RIGHT_SHOULDER, 0 = NOSE
            l_sh = landmarks[11]
            r_sh = landmarks[12]
            nose = landmarks[0]

            mid_sh_x = float((l_sh[0] + r_sh[0]) / 2.0)
            mid_sh_y = float((l_sh[1] + r_sh[1]) / 2.0)

            # Validate whether detected pose belongs to primary driver ROI
            is_driver = True
            if driver_roi_config:
                rx_min = driver_roi_config.get("x_min", 0.10)
                rx_max = driver_roi_config.get("x_max", 0.90)
                if not (rx_min <= mid_sh_x <= rx_max):
                    is_driver = False

            if driver_face_bbox and is_driver:
                fx, fy, fw, fh = driver_face_bbox
                face_cx = (fx + fw / 2.0) / float(w)
                # Ensure pose horizontal alignment is within reasonable margin of driver face
                if abs(mid_sh_x - face_cx) > 0.35:
                    is_driver = False

            if not is_driver:
                # Pose belongs to a passenger, ignore for driver fatigue
                return PostureMetrics(is_driver_pose=False, staleness_frames=self.cached_metrics.staleness_frames + 1)

            # 2. Compute Shoulder-Line Tilt Angle (Slouch / Lean)
            # dx = left_shoulder.x - right_shoulder.x (in normalized image space)
            # dy = left_shoulder.y - right_shoulder.y
            # In image coords, y increases downwards.
            dx = float(l_sh[0] - r_sh[0]) * w
            dy = float(l_sh[1] - r_sh[1]) * h
            if abs(dx) > 1e-4:
                angle_rad = np.arctan2(dy, dx)
                shoulder_angle = float(np.degrees(angle_rad))
            else:
                shoulder_angle = 0.0

            is_slouched = abs(shoulder_angle) >= self.slouch_angle_threshold

            # 3. Compute Head-to-Shoulder Vertical Droop Delta
            # Ear indices: 7 = LEFT_EAR, 8 = RIGHT_EAR
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            head_y = float((l_ear[1] + r_ear[1]) / 2.0) if (l_ear[1] > 0 and r_ear[1] > 0) else float(nose[1])
            
            # Vertical separation in normalized height: positive when head is above shoulders
            head_shoulder_distance = float(mid_sh_y - head_y)

            # Combined droop: Head pitched downward (solvePnP pitch < -18°) combined with
            # compressed head-to-shoulder vertical separation (< threshold) or severe forward slump
            is_drooping_combined = (pitch < -18.0) and (head_shoulder_distance < self.head_shoulder_droop_threshold)

            # 4. Track Movement Variance over Rolling Window (Stillness / Rigidity)
            self.shoulder_pos_history.append((mid_sh_x, mid_sh_y))
            stillness_variance = 0.0
            is_frozen_still = False

            if len(self.shoulder_pos_history) >= 30:
                pos_arr = np.array(self.shoulder_pos_history)
                var_x = float(np.var(pos_arr[:, 0]))
                var_y = float(np.var(pos_arr[:, 1]))
                stillness_variance = var_x + var_y
                if stillness_variance < self.stillness_variance_threshold and len(self.shoulder_pos_history) >= 90:
                    is_frozen_still = True

            # 5. Derive Secondary Posture Fatigue Penalty
            posture_penalty = 0.0
            if is_slouched:
                posture_penalty += 0.15 * self.posture_score_weight
            if is_drooping_combined:
                posture_penalty += 0.35 * self.posture_score_weight
            if is_frozen_still:
                posture_penalty += 0.20 * self.posture_score_weight

            self.cached_metrics = PostureMetrics(
                shoulder_angle=shoulder_angle,
                is_slouched=is_slouched,
                head_shoulder_distance=head_shoulder_distance,
                is_drooping_combined=is_drooping_combined,
                stillness_variance=stillness_variance,
                is_frozen_still=is_frozen_still,
                posture_penalty=posture_penalty,
                staleness_frames=0,
                is_driver_pose=True,
                landmarks=landmarks
            )
            return self.cached_metrics

        except Exception:
            return PostureMetrics(staleness_frames=self.cached_metrics.staleness_frames + 1)
