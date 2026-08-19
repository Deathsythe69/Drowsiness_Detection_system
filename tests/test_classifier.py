"""Unit tests for core/classifier.py FSM state transitions."""

from core.classifier import StateClassifier, State

def test_classifier_awake():
    """Verify default state is AWAKE and score is zero."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 4,
            "mar_threshold": 0.6,
            "mar_consec_frames": 4,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    state, score, events = classifier.process_frame(True, 0.30, 0.15)
    
    assert state == State.AWAKE
    assert score == 0.0
    assert len(events) == 0

def test_classifier_prolonged_closure():
    """Verify eye closure transitions AWAKE -> WARNING -> ALERT."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 4,
            "mar_threshold": 0.6,
            "mar_consec_frames": 4,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    
    # Frame 1: eye closes (ear_consec_frames / 2 is 2, so 1 frame doesn't trigger warning yet)
    state, score, events = classifier.process_frame(True, 0.10, 0.15)
    assert state == State.AWAKE
    
    # Frame 2: closed (reaches 2 frames -> triggers warning)
    state, score, events = classifier.process_frame(True, 0.10, 0.15)
    assert state == State.DROWSY_WARNING
    
    # Frame 3: closed
    state, score, events = classifier.process_frame(True, 0.10, 0.15)
    assert state == State.DROWSY_WARNING
    
    # Frame 4: closed (reaches 4 frames -> triggers alert)
    state, score, events = classifier.process_frame(True, 0.10, 0.15)
    assert state == State.DROWSY_ALERT
    assert score == 100.0
    assert "drowsy_alert" in events

def test_classifier_yawn():
    """Verify yawn detection increments fatigue score and triggers events."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 10,
            "mar_threshold": 0.6,
            "mar_consec_frames": 3,
            "fatigue_warning_score": 10.0,
            "fatigue_alert_score": 20.0
        }
    }
    classifier = StateClassifier(config)
    
    # Send 3 frames of high MAR (yawn in progress)
    classifier.process_frame(True, 0.30, 0.80)
    classifier.process_frame(True, 0.30, 0.80)
    classifier.process_frame(True, 0.30, 0.80)
    
    # Yawn is registered on mouth closing (MAR drops back below threshold)
    state, score, events = classifier.process_frame(True, 0.30, 0.15)
    
    assert "yawn" in events
    # Yawn adds 25 points, minus ~0.2 points of decay, so score should be > 20
    assert score > 20.0
    assert state == State.DROWSY_ALERT # because score 24.8 >= fatigue_alert_score 20.0

def test_classifier_face_loss():
    """Verify face loss triggers NO_FACE state after 3 seconds buffer."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 10,
            "mar_threshold": 0.6,
            "mar_consec_frames": 10,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    
    # Process initial frame with face
    state, _, _ = classifier.process_frame(True, 0.30, 0.15)
    assert state == State.AWAKE
    
    # Process frame with face lost
    state, _, _ = classifier.process_frame(False, 0.0, 0.0)
    # Should stay AWAKE initially due to 3-second grace buffer
    assert state == State.AWAKE
    
    # Simulate 4 seconds passing without face detection
    import time
    classifier.last_face_seen_time = time.time() - 4.0
    state, _, _ = classifier.process_frame(False, 0.0, 0.0)
    assert state == State.NO_FACE
