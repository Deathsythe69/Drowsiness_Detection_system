"""Unit tests for low-light preprocessing and glare mitigation."""

import numpy as np
import cv2
from core.preprocessing import calculate_brightness, apply_clahe_and_gamma, reduce_specular_glare, preprocess_frame

def test_calculate_brightness():
    """Verify luminance computation on dark, medium, and bright synthetic frames."""
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert calculate_brightness(dark_frame) == 0.0

    bright_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert 199.0 <= calculate_brightness(bright_frame) <= 201.0

def test_apply_clahe_and_gamma():
    """Verify CLAHE and Gamma brighten low-light frames."""
    dark_frame = np.full((100, 100, 3), 25, dtype=np.uint8)
    enhanced = apply_clahe_and_gamma(dark_frame, clip_limit=2.5, grid_size=8, gamma=0.5)
    
    assert enhanced.shape == dark_frame.shape
    # Enhanced frame should have higher average brightness
    assert calculate_brightness(enhanced) > calculate_brightness(dark_frame)

def test_reduce_specular_glare():
    """Verify glare reduction filter runs on hot spot images without error."""
    glare_frame = np.full((100, 100, 3), 50, dtype=np.uint8)
    # Inject high intensity glare spot
    glare_frame[40:60, 40:60] = 255
    dampened = reduce_specular_glare(glare_frame, threshold=240)
    assert dampened.shape == glare_frame.shape

def test_preprocess_frame_integration():
    """Verify preprocess_frame flags low-light and applies enhancements when enabled."""
    config = {
        "low_light": {
            "enabled": True,
            "brightness_threshold": 60.0,
            "gamma": 0.6,
            "clahe_clip_limit": 2.5,
            "clahe_grid_size": 8
        }
    }
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    out_frame, is_low_light, lum = preprocess_frame(dark_frame, config)
    
    assert is_low_light is True
    assert lum <= 30.0
    assert calculate_brightness(out_frame) > calculate_brightness(dark_frame)
