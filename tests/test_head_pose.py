"""Unit tests for solvePnP head pose estimation and posture analysis."""

import numpy as np
from core.features import calculate_head_pose, evaluate_posture_signal

def test_calculate_head_pose_none_landmarks():
    """Verify safe fallback on empty landmarks."""
    pitch, yaw, roll = calculate_head_pose(None, 640, 480)
    assert pitch == 0.0
    assert yaw == 0.0
    assert roll == 0.0

def test_evaluate_posture_signal():
    """Verify nodding and distraction posture penalties."""
    config = {
        "posture": {
            "enabled": True,
            "pitch_nod_threshold": -18.0,
            "yaw_distraction_threshold": 25.0,
            "slouch_penalty_weight": 2.0
        }
    }
    
    # 1. Normal upright posture
    is_compromised, penalty, label = evaluate_posture_signal(pitch=0.0, yaw=0.0, roll=0.0, config=config)
    assert is_compromised is False
    assert penalty == 0.0
    assert label == ""
    
    # 2. Head Drooping / Nodding (Pitch = -25 deg)
    is_compromised, penalty, label = evaluate_posture_signal(pitch=-25.0, yaw=0.0, roll=0.0, config=config)
    assert is_compromised is True
    assert penalty == 2.0
    assert label == "head_nod_droop"
    
    # 3. Looking away / Distracted (Yaw = 35 deg)
    is_compromised, penalty, label = evaluate_posture_signal(pitch=0.0, yaw=35.0, roll=0.0, config=config)
    assert is_compromised is True
    assert penalty > 0.0
    assert label == "distraction_turned_away"
