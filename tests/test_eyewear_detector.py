"""Unit tests for core/eyewear_detector.py."""

import numpy as np
import cv2
from core.eyewear_detector import EyewearDetector, EyewearType


def create_synthetic_face_mesh_landmarks():
    """Create normalized 468-point synthetic landmark array."""
    landmarks = np.zeros((468, 3), dtype=np.float32)
    
    # Left eye landmarks: 362, 385, 387, 263, 373, 380 around x=0.65, y=0.40
    for idx, (x, y) in zip([362, 385, 387, 263, 373, 380], [(0.60, 0.40), (0.63, 0.38), (0.67, 0.38), (0.70, 0.40), (0.67, 0.42), (0.63, 0.42)]):
        landmarks[idx] = [x, y, 0.0]
        
    # Right eye landmarks: 33, 160, 158, 133, 153, 144 around x=0.35, y=0.40
    for idx, (x, y) in zip([33, 160, 158, 133, 153, 144], [(0.30, 0.40), (0.33, 0.38), (0.37, 0.38), (0.40, 0.40), (0.37, 0.42), (0.33, 0.42)]):
        landmarks[idx] = [x, y, 0.0]
        
    # Cheek references: 50, 205, 280, 425 around y=0.60
    for idx, (x, y) in zip([50, 205, 280, 425], [(0.30, 0.60), (0.35, 0.60), (0.65, 0.60), (0.70, 0.60)]):
        landmarks[idx] = [x, y, 0.0]
        
    # Nose bridge: 168, 6, 197, 195
    for idx, (x, y) in zip([168, 6, 197, 195], [(0.50, 0.35), (0.50, 0.38), (0.50, 0.42), (0.50, 0.45)]):
        landmarks[idx] = [x, y, 0.0]
        
    return landmarks


def test_eyewear_detector_none_bare_eyes():
    """Verify normal bare skin and visible eyes classify as EyewearType.NONE."""
    detector = EyewearDetector({"eyewear": {"enabled": True, "hysteresis_frames": 3}})
    landmarks = create_synthetic_face_mesh_landmarks()
    
    # Create uniform skin tone frame (RGB ~ 150, 120, 100)
    frame = np.full((480, 640, 3), (100, 120, 150), dtype=np.uint8)
    
    # Process 3 frames to fill hysteresis queue
    for _ in range(3):
        ew_type, conf, meta = detector.detect_eyewear(frame, landmarks, 640, 480)
        
    assert ew_type == EyewearType.NONE
    assert conf >= 0.70
    assert meta.get("voted_type") == "NONE"


def test_eyewear_detector_sunglasses():
    """Verify dark tinted eye sockets classify as EyewearType.SUNGLASSES."""
    detector = EyewearDetector({"eyewear": {"enabled": True, "hysteresis_frames": 3, "sunglasses_lum_ratio_threshold": 0.52}})
    landmarks = create_synthetic_face_mesh_landmarks()
    
    # Skin is bright (160), but eye socket regions are dark tinted sunglasses (20)
    frame = np.full((480, 640, 3), 160, dtype=np.uint8)
    # Paint left eye socket black
    frame[int(0.35*480):int(0.45*480), int(0.58*640):int(0.72*640)] = 20
    # Paint right eye socket black
    frame[int(0.35*480):int(0.45*480), int(0.28*640):int(0.42*640)] = 20
    
    for _ in range(3):
        ew_type, conf, meta = detector.detect_eyewear(frame, landmarks, 640, 480)
        
    assert ew_type == EyewearType.SUNGLASSES
    assert meta.get("lum_ratio") < 0.50
    assert meta.get("voted_type") == "SUNGLASSES"


def test_eyewear_detector_regular_glasses_glare():
    """Verify specular reflection glints on lenses classify as EyewearType.REGULAR_GLASSES."""
    detector = EyewearDetector({"eyewear": {"enabled": True, "hysteresis_frames": 3, "glare_reflection_threshold": 218, "glare_min_pixel_ratio": 0.02}})
    landmarks = create_synthetic_face_mesh_landmarks()
    
    # Frame is normal skin (130)
    frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    
    # Add bright specular reflection hotspots (250) across both lens surfaces
    frame[int(0.38*480):int(0.42*480), int(0.62*640):int(0.66*640)] = 250
    frame[int(0.38*480):int(0.42*480), int(0.32*640):int(0.36*640)] = 250
    
    for _ in range(3):
        ew_type, conf, meta = detector.detect_eyewear(frame, landmarks, 640, 480)
        
    assert ew_type == EyewearType.REGULAR_GLASSES
    assert meta.get("glare_ratio") >= 0.02
    assert meta.get("voted_type") == "REGULAR_GLASSES"


def test_filter_glass_reflection():
    """Verify filter_glass_reflection dampens specular hotspots and returns enhanced crop."""
    # Create synthetic eye crop with a bright glare glint (255)
    eye_crop = np.full((40, 60, 3), 80, dtype=np.uint8)
    eye_crop[15:25, 25:35] = 255  # Glare spot
    
    enhanced = EyewearDetector.filter_glass_reflection(eye_crop)
    assert enhanced is not None
    assert enhanced.shape == eye_crop.shape
    
    # The max glare intensity should be softened/inpainted
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    assert float(np.mean(gray_enhanced)) > 0.0


def test_assess_eye_glare_occlusion():
    """Verify assess_eye_glare_occlusion identifies sustained glare saturation runs and returns filtered patches."""
    config = {
        "eyewear": {
            "glare_reflection_threshold": 220,
            "glare_occlusion_ratio_threshold": 0.15,
            "glare_run_frames": 3
        }
    }
    detector = EyewearDetector(config)
    
    # 1. Test clean eye crops (no glare)
    crop_l = np.full((30, 40, 3), 70, dtype=np.uint8)
    crop_r = np.full((30, 40, 3), 70, dtype=np.uint8)
    is_run, unknown, ratio, filt_l, filt_r = detector.assess_eye_glare_occlusion(crop_l, crop_r)
    assert not is_run
    assert not unknown
    assert ratio == 0.0
    assert detector.consecutive_glare_frames == 0
    
    # 2. Test severe glare saturation (>30% of pixels >= 240)
    glare_l = np.full((30, 40, 3), 70, dtype=np.uint8)
    glare_r = np.full((30, 40, 3), 70, dtype=np.uint8)
    glare_l[5:25, 10:30] = 250
    glare_r[5:25, 10:30] = 250
    
    # Frame 1: Glare starts, but run length is 3 so is_run is not yet true
    is_run, unknown, ratio, _, _ = detector.assess_eye_glare_occlusion(glare_l, glare_r)
    assert not is_run
    assert ratio > 0.15
    assert detector.consecutive_glare_frames == 1
    
    # Frame 2:
    is_run, unknown, ratio, _, _ = detector.assess_eye_glare_occlusion(glare_l, glare_r)
    assert not is_run
    assert detector.consecutive_glare_frames == 2
    
    # Frame 3: Sustained glare run reached!
    is_run, unknown, ratio, filt_l, filt_r = detector.assess_eye_glare_occlusion(glare_l, glare_r)
    assert is_run
    assert filt_l is not None
    assert filt_r is not None

