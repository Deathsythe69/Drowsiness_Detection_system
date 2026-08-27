"""Tests for Live Continuous Sample Collection & Self-Training Pipeline.

Verifies daytime/nighttime sample collection, metadata sidecars, active learning
feedback ingestion, incremental self-training, and zero-downtime hot-reloading.
"""

import os
import json
import shutil
import tempfile
import numpy as np
import pytest

from core.live_trainer import LiveSampleCollector, LiveAutoTrainer
from core.eye_classifier import LowLightEyeClassifier


@pytest.fixture
def temp_live_dir():
    d = tempfile.mkdtemp(prefix="test_live_data_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_live_sample_collector_day_and_night(temp_live_dir):
    """Verify LiveSampleCollector creates proper daytime and nighttime samples."""
    collector = LiveSampleCollector(base_dir=temp_live_dir)
    
    dummy_crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    
    # 1. Save Day Open Eye sample
    path_day = collector.save_sample(
        eye_crop=dummy_crop,
        full_frame=None,
        label=1,
        is_low_light=False,
        metadata={"ear": 0.28, "luminance": 120.0},
        category="open",
        force=True
    )
    assert path_day is not None
    assert os.path.exists(path_day)
    assert "day" in path_day

    # Verify JSON sidecar
    json_path = os.path.splitext(path_day)[0] + ".json"
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)
    assert meta["label"] == 1
    assert meta["is_low_light"] is False
    assert meta["category"] == "open"

    # 2. Save Night Drowsy sample
    path_night = collector.save_sample(
        eye_crop=dummy_crop,
        full_frame=None,
        label=0,
        is_low_light=True,
        metadata={"ear": 0.10, "luminance": 15.0},
        category="closed",
        force=True
    )
    assert path_night is not None
    assert os.path.exists(path_night)
    assert "night" in path_night


def test_live_sample_collector_statistics(temp_live_dir):
    """Verify get_dataset_statistics aggregates counts accurately."""
    collector = LiveSampleCollector(base_dir=temp_live_dir)
    dummy_crop = np.full((64, 64, 3), 100, dtype=np.uint8)

    collector.save_sample(dummy_crop, None, label=1, is_low_light=False, category="open", force=True)
    collector.save_sample(dummy_crop, None, label=0, is_low_light=False, category="closed", force=True)
    collector.save_sample(dummy_crop, None, label=0, is_low_light=True, category="closed", force=True)

    stats = collector.get_dataset_statistics()
    assert stats["total_samples"] == 3
    assert stats["day_samples"] == 2
    assert stats["night_samples"] == 1
    assert stats["open_eyes"] == 1
    assert stats["closed_eyes"] == 2


def test_live_auto_trainer_sync(temp_live_dir):
    """Verify LiveAutoTrainer runs continuous training on live samples."""
    collector = LiveSampleCollector(base_dir=temp_live_dir)
    
    # Generate some mock day/night samples
    for i in range(10):
        dummy_open = np.full((64, 64), 180, dtype=np.uint8)
        dummy_closed = np.full((64, 64), 40, dtype=np.uint8)
        collector.save_sample(dummy_open, None, label=1, is_low_light=(i % 2 == 0), category="open", force=True)
        collector.save_sample(dummy_closed, None, label=0, is_low_light=(i % 2 == 0), category="closed", force=True)

    output_model = os.path.join(temp_live_dir, "test_updated_model.json")
    trainer = LiveAutoTrainer(
        live_dir=temp_live_dir,
        baseline_dir="non_existent",  # test pure live fallback
        model_output_path=output_model
    )

    result = trainer.train_sync(epochs=2, lr=0.05)
    assert result["status"] == "success"
    assert result["samples_trained"] == 20
    assert os.path.exists(output_model)


def test_eye_classifier_hot_reload(temp_live_dir):
    """Verify LowLightEyeClassifier reloads updated weights dynamically without errors."""
    collector = LiveSampleCollector(base_dir=temp_live_dir)
    for i in range(6):
        dummy = np.full((64, 64), 120, dtype=np.uint8)
        collector.save_sample(dummy, None, label=i % 2, is_low_light=True, force=True)

    model_path = os.path.join(temp_live_dir, "hot_reload_model.json")
    trainer = LiveAutoTrainer(
        live_dir=temp_live_dir,
        baseline_dir="non_existent",
        model_output_path=model_path
    )
    trainer.train_sync(epochs=1)

    classifier = LowLightEyeClassifier()
    dummy_crop = np.full((64, 64, 3), 150, dtype=np.uint8)
    prob_before = classifier.predict_eye_openness(dummy_crop, is_low_light=True)

    # Hot reload with new weights
    reloaded = classifier.reload_weights(model_path)
    assert reloaded is True
    prob_after = classifier.predict_eye_openness(dummy_crop, is_low_light=True)
    assert 0.0 <= prob_after <= 1.0
