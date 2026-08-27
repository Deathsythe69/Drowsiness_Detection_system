"""Tests for Sweet Spot Multi-Feature Fusion (EAR, MAR, Pitch, Yaw).

Verifies that isolated driving actions (looking down at navigation, checking mirrors,
speaking) do NOT trigger false alarms when eyes are open, while true compound
drowsiness and sustained eye closure reliably trigger alerts.
"""

import pytest
from core.classifier import StateClassifier, State


@pytest.fixture
def classifier():
    config = {
        "thresholds": {
            "ear_threshold": 0.16,
            "ear_consec_frames": 25,
            "mar_threshold": 0.65,
            "mar_consec_frames": 20,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        },
        "posture": {
            "enabled": True,
            "pitch_nod_threshold": -22.0,
            "yaw_distraction_threshold": 30.0,
            "slouch_penalty_weight": 0.20
        }
    }
    return StateClassifier(config)


def test_looking_down_with_open_eyes_does_not_sound_alarm(classifier):
    """Driver glances down at dashboard/speedometer with open eyes for 60 frames."""
    for _ in range(60):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.28,
            mar=0.15,
            pitch=-25.0,  # Looking down
            yaw=0.0,
            roll=0.0,
            eye_open_prob=0.92  # Eyes clearly open
        )
    
    # Looking down alone with open eyes must NOT sound the buzzer
    assert state != State.DROWSY_ALERT
    assert score < 40.0


def test_mirror_check_glance_with_open_eyes_does_not_sound_alarm(classifier):
    """Driver turns head to check side mirrors (high yaw) with eyes open."""
    for _ in range(45):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.26,
            mar=0.15,
            pitch=0.0,
            yaw=35.0,  # Checking right mirror
            roll=0.0,
            eye_open_prob=0.88
        )
    
    assert state != State.DROWSY_ALERT
    assert score < 35.0


def test_talking_mouth_movement_does_not_trigger_yawn(classifier):
    """Driver is speaking or singing (moderate MAR fluctuations, not a yawning stretch)."""
    for _ in range(30):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.28,
            mar=0.45,  # Talking
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            eye_open_prob=0.95
        )
    
    assert "yawn" not in events
    assert state == State.AWAKE
    assert score == 0.0


def test_compound_drowsiness_synergy(classifier):
    """Drooping eyes combined with downward nodding triggers compound drowsiness escalation."""
    compound_triggered = False
    for _ in range(30):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.15,  # Drooping eye
            mar=0.15,
            pitch=-24.0,  # Nodding downward
            yaw=0.0,
            roll=0.0,
            eye_open_prob=0.40
        )
        if "compound_drowsiness" in events:
            compound_triggered = True

    assert compound_triggered is True
    assert score > 0.0


def test_sustained_eye_closure_triggers_alert(classifier):
    """True closed eyes (microsleep) for 25 consecutive frames triggers DROWSY_ALERT."""
    for i in range(25):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.08,
            mar=0.15,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            eye_open_prob=0.05
        )
    
    assert state == State.DROWSY_ALERT
    assert score == 100.0
