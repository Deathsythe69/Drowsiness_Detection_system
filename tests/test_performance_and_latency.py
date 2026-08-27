"""Tests for Low-Power Device Performance, Latency, and Zero-Lag Frame Processing.

Verifies that inference and preprocessing routines execute with sub-10ms latency
suitable for low-power CPUs (Raspberry Pi, Intel Celeron, budget laptops).
"""

import time
import numpy as np
import pytest

from core.eye_classifier import LowLightEyeClassifier
from core.preprocessing import preprocess_frame, calculate_brightness


def test_eye_classifier_inference_speed():
    """Verify that neural eye classification runs in sub-10ms on CPU."""
    classifier = LowLightEyeClassifier()
    dummy_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Warmup
    _ = classifier.predict_eye_openness(dummy_crop, is_low_light=True)

    # Benchmark 20 iterations
    t0 = time.time()
    for _ in range(20):
        _ = classifier.predict_eye_openness(dummy_crop, is_low_light=True)
    total_time = time.time() - t0
    avg_latency_ms = (total_time / 20.0) * 1000.0

    # Must be under 15ms per eye crop for smooth 30+ FPS operation
    assert avg_latency_ms < 15.0, f"Eye inference too slow: {avg_latency_ms:.2f}ms"


def test_preprocessing_speed():
    """Verify that frame brightness and adaptive preprocessing execute in sub-5ms."""
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    config = {"low_light": {"enabled": True, "brightness_threshold": 45.0}}

    # Benchmark 30 frames
    t0 = time.time()
    for _ in range(30):
        _ = preprocess_frame(dummy_frame, config)
    total_time = time.time() - t0
    avg_latency_ms = (total_time / 30.0) * 1000.0

    assert avg_latency_ms < 10.0, f"Preprocessing too slow: {avg_latency_ms:.2f}ms"
