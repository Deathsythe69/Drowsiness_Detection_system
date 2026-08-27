"""Unit tests for core/classifier.py FSM state transitions."""

from core.classifier import StateClassifier, State

def _skip_calibration(classifier):
    """Bypass calibration phase for unit tests that test FSM logic directly."""
    classifier.is_calibrating = False
    classifier.is_calibrated = True
    return classifier

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
    _skip_calibration(classifier)
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
    _skip_calibration(classifier)
    
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
    assert score >= 75.0
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
    _skip_calibration(classifier)
    
    # Send 3 frames of high MAR (yawn in progress)
    classifier.process_frame(True, 0.30, 0.80)
    classifier.process_frame(True, 0.30, 0.80)
    classifier.process_frame(True, 0.30, 0.80)
    
    # Yawn is registered on mouth closing (MAR drops back below threshold)
    state, score, events = classifier.process_frame(True, 0.30, 0.15)
    
    assert "yawn" in events
    # Yawn adds 15 points, minus decay, so score should be >= warning threshold 10.0
    assert score >= 10.0
    assert state in (State.DROWSY_WARNING, State.DROWSY_ALERT)

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
    _skip_calibration(classifier)
    
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

def test_classifier_perclos_accumulation():
    """Verify automotive PERCLOS (Percentage of Eye Closure) calculates accurately and triggers alert."""
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 20, # high consecutive frame threshold so micro-blinks don't trigger alert directly
            "max_blink_frames": 15,
            "perclos_threshold": 0.35, # 35% eye closure triggers fatigue
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0,
            "perclos_window_short_seconds": 2, # 60 frames at 30fps
            "perclos_window_long_seconds": 10
        },
        "performance": {
            "target_fps": 30
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # Process 60 frames: 35 closed (short bursts below max_blink_frames), 25 open
    # Closed ratio in rolling 60 frames = 35 / 60 ≈ 58.3% > 35% threshold
    for i in range(60):
        # alternate: 7 closed, 5 open, etc. (each burst < 15 frames max_blink_frames)
        is_closed = (i % 12) < 7
        ear = 0.10 if is_closed else 0.30
        state, score, events = classifier.process_frame(True, ear, 0.15)
        
    assert classifier.perclos_60 > 0.35
    assert classifier.fatigue_score > 0.0


def test_classifier_sunglasses_bypasses_ear_closure():
    """Verify that dark sunglasses with 0.0 EAR do NOT trigger false eye closure alarms."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 5,
            "mar_threshold": 0.60,
            "mar_consec_frames": 5,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        },
        "posture": {
            "enabled": True,
            "pitch_nod_threshold": -22.0,
            "yaw_distraction_threshold": 30.0
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # Process 20 consecutive frames with EAR = 0.0 (simulating pitch-black sunglasses lenses)
    # Pitch is upright (0.0), MAR is normal mouth closed (0.15)
    for _ in range(20):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.0,
            mar=0.15,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            eyewear_type=EyewearType.SUNGLASSES
        )
        
    assert state == State.AWAKE
    assert classifier.consec_eye_closed == 0
    assert score == 0.0


def test_classifier_sunglasses_yawn_alert():
    """Verify that in Sunglasses mode, yawning (MAR) triggers fatigue warnings and alerts."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 10,
            "mar_threshold": 0.60,
            "mar_consec_frames": 3,
            "yawn_window_seconds": 300,
            "yawn_frequency_alert_threshold": 3,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # Simulate Yawn 1
    for _ in range(4):
        classifier.process_frame(True, 0.0, 0.80, pitch=0.0, eyewear_type=EyewearType.SUNGLASSES)
    state, score, events = classifier.process_frame(True, 0.0, 0.15, pitch=0.0, eyewear_type=EyewearType.SUNGLASSES)
    assert "yawn" in events
    assert score >= 34.0
    
    # Simulate Yawn 2 (Rolling count reaches 2 -> triggers warning)
    for _ in range(4):
        classifier.process_frame(True, 0.0, 0.80, pitch=0.0, eyewear_type=EyewearType.SUNGLASSES)
    state, score, events = classifier.process_frame(True, 0.0, 0.15, pitch=0.0, eyewear_type=EyewearType.SUNGLASSES)
    assert classifier.active_yawn_count == 2
    assert state in (State.DROWSY_WARNING, State.DROWSY_ALERT)


def test_classifier_sunglasses_head_nod_alert():
    """Verify that in Sunglasses mode, downward head nodding triggers fatigue accumulation and alerts."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 10,
            "mar_threshold": 0.60,
            "mar_consec_frames": 5,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        },
        "posture": {
            "enabled": True,
            "pitch_nod_threshold": -20.0,
            "yaw_distraction_threshold": 30.0
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # Simulate driver nodding downward (Pitch = -28.0°) for 35 frames
    for _ in range(35):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.0,
            mar=0.15,
            pitch=-28.0,
            yaw=0.0,
            eyewear_type=EyewearType.SUNGLASSES
        )
        
    assert state == State.DROWSY_ALERT
    assert score >= 75.0
    assert "drowsy_alert" in events
    assert "sunglasses_mode" in events


def test_classifier_glare_occlusion_freezes_perclos():
    """Verify that when eye_state_unknown / glare occlusion occurs, corrupted EAR is NOT added to PERCLOS."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 25,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0,
            "perclos_window_short_seconds": 1,
            "perclos_window_long_seconds": 2
        },
        "performance": {
            "target_fps": 10
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # 1. Feed 10 closed eye frames to establish a non-zero PERCLOS
    for _ in range(10):
        classifier.process_frame(True, ear=0.10, mar=0.15)
        
    initial_perclos = classifier.perclos_60
    assert initial_perclos > 0.0
    initial_score = classifier.fatigue_score
    initial_deque_len = len(classifier.perclos_window_short)
    
    # 2. Feed 10 glare-occluded frames with fake 'open' EAR=0.35 but eye_state_unknown=True
    for _ in range(10):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.35,  # corrupted glare EAR
            mar=0.15,
            eyewear_type=EyewearType.REGULAR_GLASSES,
            eye_state_unknown=True,
            is_glare_occluded=True
        )
        assert "eye_region_glare" in events
        
    # PERCLOS window should NOT have appended 10 fake 0.0s (deque length unchanged)
    assert len(classifier.perclos_window_short) == initial_deque_len
    # Fatigue score should NOT have decayed back to 0.0
    assert classifier.fatigue_score >= initial_score


def test_classifier_glare_prolonged_blindness_warning():
    """Verify that prolonged glare blindness (>= 60 frames) escalates safety caution up to warning level."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.21,
            "ear_consec_frames": 25,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        },
        "eyewear": {
            "glare_max_blind_frames": 20
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)
    
    # Feed 25 consecutive glare-blinded frames
    for _ in range(25):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.30,
            mar=0.15,
            eyewear_type=EyewearType.REGULAR_GLASSES,
            eye_state_unknown=True,
            is_glare_occluded=True
        )
        
    assert "eye_glare_blindness_warning" in events
    assert classifier.consec_glare_frames == 25
    assert classifier.fatigue_score > 0.0


def test_classifier_calibration_flow():
    """Verify that startup calibration collects baseline and sets personalized EAR threshold."""
    config = {
        "thresholds": {
            "ear_threshold": 0.16,
            "calibration_frames": 10,
            "baseline_ear_ratio": 0.70,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    assert classifier.is_calibrating is True
    assert classifier.is_calibrated is False

    # Feed 9 frames (still calibrating)
    for _ in range(9):
        state, score, events = classifier.process_frame(True, ear=0.30, mar=0.15)
        assert state == State.AWAKE
        assert score == 0.0
        assert "calibrating" in events

    # Frame 10 completes calibration
    state, score, events = classifier.process_frame(True, ear=0.30, mar=0.15)
    assert classifier.is_calibrating is False
    assert classifier.is_calibrated is True
    assert "calibration_complete" in events
    assert classifier.baseline_ear == 0.30
    assert abs(classifier.ear_threshold - (0.30 * 0.70)) < 1e-4


def test_classifier_glasses_glare_ema_guard_and_reduced_confidence():
    """Verify corrupted EAR during glare does not poison EMA or cause false eye-closed alerts."""
    from core.eyewear_detector import EyewearType
    config = {
        "thresholds": {
            "ear_threshold": 0.20,
            "ear_consec_frames": 5,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)

    # Establish clean baseline with open eyes
    classifier.process_frame(True, ear=0.30, mar=0.15)
    clean_ema_ear = classifier.ema_ear
    assert clean_ema_ear is not None and clean_ema_ear >= 0.28

    # Driver puts on glasses causing specular reflection / glare (ear reported as 0.0)
    for _ in range(10):
        state, score, events = classifier.process_frame(
            has_face=True,
            ear=0.0,  # Corrupted EAR from glare/reflection
            mar=0.15,
            eyewear_type=EyewearType.REGULAR_GLASSES,
            eye_state_unknown=True,
            is_glare_occluded=True
        )
        assert "eye_region_glare" in events
        assert "reduced_confidence" in events
        # Must not immediately trigger drowsy alert
        assert state != State.DROWSY_ALERT

    # EMA EAR should NOT have been corrupted down to 0.0
    assert classifier.ema_ear == clean_ema_ear


def test_classifier_alarm_latching_and_admin_reset():
    """Verify once DROWSY_ALERT triggers, alarm latches until explicit admin_reset()."""
    config = {
        "thresholds": {
            "ear_threshold": 0.20,
            "ear_consec_frames": 4,
            "fatigue_warning_score": 40.0,
            "fatigue_alert_score": 75.0
        }
    }
    classifier = StateClassifier(config)
    _skip_calibration(classifier)

    # Trigger DROWSY_ALERT via sustained eye closure
    for _ in range(4):
        state, score, events = classifier.process_frame(True, ear=0.08, mar=0.15)

    assert state == State.DROWSY_ALERT
    assert classifier.alarm_latched is True
    assert "drowsy_alert" in events

    # Driver opens eyes wide (EAR 0.35)
    for _ in range(10):
        state, score, events = classifier.process_frame(True, ear=0.35, mar=0.15)
        # State MUST stay locked in DROWSY_ALERT due to safety latch
        assert state == State.DROWSY_ALERT
        assert classifier.alarm_latched is True
        assert "alarm_latched" in events

    # Explicit admin_reset clears the latch and resets fatigue
    classifier.admin_reset()
    assert classifier.alarm_latched is False
    _skip_calibration(classifier)

    state, score, events = classifier.process_frame(True, ear=0.35, mar=0.15)
    assert state == State.AWAKE
    assert score == 0.0



