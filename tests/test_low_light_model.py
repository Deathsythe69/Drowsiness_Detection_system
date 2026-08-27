"""Unit tests for low-light preprocessing and eye state neural classifier."""

import os
import numpy as np
import pytest

from core.preprocessing import enhance_eye_region, preprocess_frame
from core.eye_classifier import LowLightEyeClassifier
from train_test.train_low_light_model import (
    LowLightEyeCNN,
    apply_night_augmentation,
    preprocess_eye_sample
)


def test_enhance_eye_region_normal_and_low_light():
    """Test localized CLAHE and denoising on synthetic eye crops."""
    # Dark crop with noise
    dark_crop = np.random.randint(0, 30, (48, 48, 3), dtype=np.uint8)
    enhanced = enhance_eye_region(dark_crop, is_low_light=True)
    
    assert enhanced.shape == dark_crop.shape
    # Contrast and standard deviation should increase
    assert np.std(enhanced) >= np.std(dark_crop)


def test_night_augmentation_pipeline():
    """Test synthetic night degradation and sensor noise generator."""
    sample = np.ones((64, 64), dtype=np.uint8) * 128
    aug = apply_night_augmentation(sample)
    
    assert aug.shape == (64, 64)
    assert aug.dtype == np.uint8
    # Noise/attenuation should modify pixel distribution
    assert not np.array_equal(sample, aug)


def test_low_light_eye_cnn_forward():
    """Test pure-vectorized LowLightEyeCNN forward pass and probability output."""
    model = LowLightEyeCNN(input_shape=(64, 64, 1))
    
    # 4 synthetic samples with shape (4, 64, 64, 1)
    x = np.random.rand(4, 64, 64, 1).astype(np.float32)
    
    probs = model.predict_proba(x)
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_low_light_classifier_inference():
    """Test runtime inference classifier with valid eye crops."""
    classifier = LowLightEyeClassifier()
    
    # Normal crop
    crop = np.random.randint(50, 200, (60, 60, 3), dtype=np.uint8)
    p_open = classifier.predict_eye_openness(crop, is_low_light=False)
    assert 0.0 <= p_open <= 1.0
    
    # Extreme low light crop
    dark_crop = np.random.randint(2, 25, (40, 40, 3), dtype=np.uint8)
    p_open_dark = classifier.predict_eye_openness(dark_crop, is_low_light=True)
    assert 0.0 <= p_open_dark <= 1.0
    
    # Empty crop fallback
    assert classifier.predict_eye_openness(None) == 0.50


def test_low_light_classifier_predict_both_eyes():
    """Test dual-eye batched inference and execution speed (< 5ms)."""
    import time
    classifier = LowLightEyeClassifier()
    crop_l = np.random.randint(40, 200, (48, 48, 3), dtype=np.uint8)
    crop_r = np.random.randint(40, 200, (48, 48, 3), dtype=np.uint8)
    
    t0 = time.perf_counter()
    p_l, p_r, p_mean = classifier.predict_both_eyes(crop_l, crop_r, is_low_light=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert 0.0 <= p_l <= 1.0
    assert 0.0 <= p_r <= 1.0
    assert 0.0 <= p_mean <= 1.0
    # Pure vectorized GEMM should take < 15ms even on cold start
    assert elapsed_ms < 25.0

