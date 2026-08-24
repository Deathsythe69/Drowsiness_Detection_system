"""Unit tests for rolling time window yawn frequency tracking and state escalation."""

from core.classifier import StateClassifier, State

def test_yawn_frequency_escalation():
    """Verify that multiple yawns in a rolling window escalate fatigue and trigger alert."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 20,
            "mar_threshold": 0.60,
            "mar_consec_frames": 5,  # short for testing
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0,
            "yawn_window_seconds": 300,
            "yawn_frequency_alert_threshold": 3
        },
        "posture": {
            "enabled": True
        }
    }
    
    classifier = StateClassifier(config)
    
    def simulate_single_yawn():
        # Yawn opening for 6 frames
        for _ in range(6):
            classifier.process_frame(has_face=True, ear=0.35, mar=0.75)
        # Mouth closes
        classifier.process_frame(has_face=True, ear=0.35, mar=0.15)
        
    # Yawn 1: should register yawn event
    simulate_single_yawn()
    assert classifier.active_yawn_count == 1
    
    # Yawn 2
    simulate_single_yawn()
    assert classifier.active_yawn_count == 2
    
    # Yawn 3: crosses yawn_frequency_alert_threshold -> triggers escalation and elevates to warning/alert
    simulate_single_yawn()
    assert classifier.active_yawn_count == 3
    state, score, events = classifier.process_frame(has_face=True, ear=0.35, mar=0.15)
    
    assert state in [State.DROWSY_WARNING, State.DROWSY_ALERT]
    assert score >= 40.0
    
    # Yawn 4: crosses alert threshold (>= limit + 1) -> triggers DROWSY_ALERT
    simulate_single_yawn()
    assert classifier.active_yawn_count == 4
    state, score, events = classifier.process_frame(has_face=True, ear=0.35, mar=0.15)
    assert state == State.DROWSY_ALERT
