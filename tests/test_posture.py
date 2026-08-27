"""Unit tests for Full-Body Posture Module (MediaPipe Pose).

Tests:
1. PostureDetector initialization and configuration loading.
2. Shoulder-line angle calculation for level vs slouched shoulders.
3. Head-to-shoulder droop delta calculation with pitch combination.
4. Rolling movement variance and rigid stillness detection.
5. Multi-person driver ROI filtering (passenger pose rejection).
6. Classifier secondary posture fusion without standalone alert triggers.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from core.posture import PostureDetector, PostureMetrics
from core.classifier import StateClassifier, State
from core.eyewear_detector import EyewearType


def create_synthetic_pose_landmarks(
    nose=(0.5, 0.25, 0.0),
    left_ear=(0.55, 0.25, 0.0),
    right_ear=(0.45, 0.25, 0.0),
    left_shoulder=(0.60, 0.50, 0.0),
    right_shoulder=(0.40, 0.50, 0.0),
    left_hip=(0.58, 0.85, 0.0),
    right_hip=(0.42, 0.85, 0.0)
) -> MagicMock:
    """Helper to create a mock MediaPipe Pose results object with 33 landmarks."""
    landmarks_list = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(33)]
    
    # 0 = NOSE
    landmarks_list[0].x, landmarks_list[0].y, landmarks_list[0].z = nose
    # 7 = LEFT_EAR, 8 = RIGHT_EAR
    landmarks_list[7].x, landmarks_list[7].y, landmarks_list[7].z = left_ear
    landmarks_list[8].x, landmarks_list[8].y, landmarks_list[8].z = right_ear
    # 11 = LEFT_SHOULDER, 12 = RIGHT_SHOULDER
    landmarks_list[11].x, landmarks_list[11].y, landmarks_list[11].z = left_shoulder
    landmarks_list[12].x, landmarks_list[12].y, landmarks_list[12].z = right_shoulder
    # 23 = LEFT_HIP, 24 = RIGHT_HIP
    landmarks_list[23].x, landmarks_list[23].y, landmarks_list[23].z = left_hip
    landmarks_list[24].x, landmarks_list[24].y, landmarks_list[24].z = right_hip
    
    results = MagicMock()
    results.pose_landmarks = MagicMock()
    results.pose_landmarks.landmark = landmarks_list
    return results


class TestPostureDetector:
    """Test suite for PostureDetector algorithms and signals."""

    @pytest.fixture
    def sample_config(self):
        return {
            "posture": {
                "enabled": True,
                "slouch_angle_threshold": 12.0,
                "head_shoulder_droop_threshold": 0.18,
                "stillness_window_seconds": 10,
                "stillness_variance_threshold": 0.00015,
                "posture_score_weight": 1.0,
                "pose_model_complexity": 0,
                "pose_inference_interval_frames": 1
            },
            "thresholds": {
                "ear_threshold": 0.16,
                "ear_consec_frames": 25,
                "mar_threshold": 0.65,
                "mar_consec_frames": 20,
                "fatigue_warning_score": 40.0,
                "fatigue_alert_score": 75.0
            }
        }

    def test_initialization_defaults(self, sample_config):
        detector = PostureDetector(sample_config)
        assert detector.enabled is True
        assert detector.slouch_angle_threshold == 12.0
        assert detector.head_shoulder_droop_threshold == 0.18
        assert detector.stillness_variance_threshold == 0.00015
        assert detector.model_complexity == 0

    def test_level_shoulders_not_slouched(self, sample_config):
        detector = PostureDetector(sample_config)
        
        # Level horizontal shoulders: Left (0.60, 0.50), Right (0.40, 0.50)
        mock_results = create_synthetic_pose_landmarks(
            left_shoulder=(0.60, 0.50, 0.0),
            right_shoulder=(0.40, 0.50, 0.0)
        )
        detector.pose.process = MagicMock(return_value=mock_results)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = detector.process_frame(frame, pitch=0.0, force_inference=True)
        
        assert abs(metrics.shoulder_angle) < 1.0
        assert metrics.is_slouched is False
        assert metrics.is_driver_pose is True

    def test_tilted_shoulders_slouch_detected(self, sample_config):
        detector = PostureDetector(sample_config)
        
        # Tilted shoulders: Left is higher in image (lower y) or lower (higher y)
        # Left (0.60, 0.58), Right (0.40, 0.42) -> dy = (0.58-0.42)*480 = 76.8, dx = 0.20*640 = 128
        # arctan2(76.8, 128) ≈ 30.9 degrees > 12.0 degrees
        mock_results = create_synthetic_pose_landmarks(
            left_shoulder=(0.60, 0.58, 0.0),
            right_shoulder=(0.40, 0.42, 0.0)
        )
        detector.pose.process = MagicMock(return_value=mock_results)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = detector.process_frame(frame, pitch=0.0, force_inference=True)
        
        assert metrics.is_slouched is True
        assert abs(metrics.shoulder_angle) > 12.0
        assert metrics.posture_penalty >= 0.15

    def test_head_to_shoulder_droop_collapse(self, sample_config):
        detector = PostureDetector(sample_config)
        
        # Head collapsing down into chest:
        # Ears at y=0.46, Shoulders at y=0.50 -> vertical separation = 0.04 < 0.18 threshold
        mock_results = create_synthetic_pose_landmarks(
            nose=(0.50, 0.47, 0.0),
            left_ear=(0.55, 0.46, 0.0),
            right_ear=(0.45, 0.46, 0.0),
            left_shoulder=(0.60, 0.50, 0.0),
            right_shoulder=(0.40, 0.50, 0.0)
        )
        detector.pose.process = MagicMock(return_value=mock_results)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # When pitch is downward (nodding: -24.0 deg) and vertical separation is compressed
        metrics = detector.process_frame(frame, pitch=-24.0, force_inference=True)
        assert metrics.head_shoulder_distance < 0.18
        assert metrics.is_drooping_combined is True
        assert metrics.posture_penalty >= 0.35

    def test_rolling_stillness_variance(self, sample_config):
        detector = PostureDetector(sample_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 1. Simulate rigid frozen driver across 100 consecutive frames with identical coords
        mock_frozen = create_synthetic_pose_landmarks(
            left_shoulder=(0.60, 0.50, 0.0),
            right_shoulder=(0.40, 0.50, 0.0)
        )
        detector.pose.process = MagicMock(return_value=mock_frozen)
        
        for _ in range(95):
            metrics = detector.process_frame(frame, pitch=0.0, force_inference=True)
            
        assert metrics.stillness_variance < 1e-6
        assert metrics.is_frozen_still is True

    def test_driver_roi_filtering_rejects_passenger(self, sample_config):
        detector = PostureDetector(sample_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Pose located on far right of vehicle (x = 0.95) outside driver ROI (x_min=0.10, x_max=0.80)
        mock_passenger = create_synthetic_pose_landmarks(
            left_shoulder=(0.98, 0.50, 0.0),
            right_shoulder=(0.92, 0.50, 0.0)
        )
        detector.pose.process = MagicMock(return_value=mock_passenger)
        
        roi_cfg = {"x_min": 0.10, "x_max": 0.80, "y_min": 0.05, "y_max": 0.95}
        metrics = detector.process_frame(frame, pitch=0.0, driver_roi_config=roi_cfg, force_inference=True)
        
        assert metrics.is_driver_pose is False

    def test_classifier_secondary_posture_fusion_not_standalone(self, sample_config):
        """Verify posture penalties are secondary and do not independently trigger DROWSY_ALERT."""
        classifier = StateClassifier(sample_config)
        classifier.is_calibrating = False
        classifier.is_calibrated = True
        
        # Slouched posture metrics
        slouch_metrics = PostureMetrics(
            shoulder_angle=25.0,
            is_slouched=True,
            head_shoulder_distance=0.10,
            is_drooping_combined=True,
            stillness_variance=0.00001,
            is_frozen_still=True,
            posture_penalty=0.70
        )
        
        # Frame with wide open eyes (EAR 0.30) and no yawns (MAR 0.20)
        for _ in range(10):
            state, score, events = classifier.process_frame(
                has_face=True,
                ear=0.30,
                mar=0.20,
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                eye_open_prob=1.0,
                posture_metrics=slouch_metrics
            )
            
        # Should NOT trigger DROWSY_ALERT (must remain AWAKE or capped warning level)
        assert state in (State.AWAKE, State.DROWSY_WARNING)
        assert state != State.DROWSY_ALERT
        assert score <= 35.0
        assert "shoulder_slouch" in events or "torso_head_droop_collapse" in events or "posture_rigidity_stillness" in events
