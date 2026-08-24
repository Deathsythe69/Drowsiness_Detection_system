"""Unit tests for Vehicle Motion Detector module."""

import numpy as np
import cv2
from core.motion_detector import VehicleMotionDetector


def test_motion_detector_init():
    """Verify VehicleMotionDetector initialization with default and custom parameters."""
    detector = VehicleMotionDetector(motion_threshold=5.0, window_size=10, min_moving_ratio=0.6)
    assert detector.motion_threshold == 5.0
    assert detector.window_size == 10
    assert detector.min_moving_ratio == 0.6
    assert detector.is_moving is False
    assert detector.motion_score == 0.0


def test_motion_detector_none_or_empty_frame():
    """Verify handling of None and empty frames."""
    detector = VehicleMotionDetector()
    is_moving, score = detector.update(None)
    assert is_moving is False
    assert score == 0.0

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    is_moving, score = detector.update(empty_frame)
    assert is_moving is False
    assert score == 0.0


def test_motion_detector_static_frames():
    """Verify that a sequence of static identical frames reports vehicle as stationary/parked."""
    detector = VehicleMotionDetector(motion_threshold=3.0, window_size=5, min_moving_ratio=0.4)
    static_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    for _ in range(10):
        is_moving, score = detector.update(static_frame)

    assert is_moving is False
    assert score < 1.0


def test_motion_detector_moving_scenery():
    """Verify that dynamic changes in peripheral regions trigger is_moving = True."""
    detector = VehicleMotionDetector(motion_threshold=2.0, window_size=5, min_moving_ratio=0.4)

    # Alternate background noise in the peripheral regions to simulate moving road/sky
    for i in range(10):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Add high-contrast patterns along top and bottom strips
        if i % 2 == 0:
            frame[0:80, :] = 255
            frame[400:480, :] = 20
        else:
            frame[0:80, :] = 20
            frame[400:480, :] = 255
        is_moving, score = detector.update(frame)

    assert is_moving is True
    assert score > 2.0


def test_motion_detector_reset():
    """Verify that reset clears motion history and state."""
    detector = VehicleMotionDetector(motion_threshold=2.0, window_size=5)

    for i in range(8):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detector.update(frame)

    assert detector.motion_score > 0.0
    detector.reset()
    assert detector.is_moving is False
    assert detector.motion_score == 0.0
