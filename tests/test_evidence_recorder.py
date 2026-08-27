"""Unit and integration tests for the Evidence Video Recording System."""

import os
import time
import shutil
import tempfile
import numpy as np
import pytest

from core.evidence_recorder import EvidenceRecorder


@pytest.fixture
def temp_evidence_dir():
    """Create a temporary directory for evidence recordings."""
    temp_dir = tempfile.mkdtemp(prefix="test_evidence_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_recorder_initialization(temp_evidence_dir):
    """Test recorder initializes directory and parameters properly."""
    recorder = EvidenceRecorder(
        output_dir=temp_evidence_dir,
        pre_buffer_seconds=1.0,
        post_buffer_seconds=1.0,
        fps=10.0,
        cooldown_seconds=2.0
    )
    assert os.path.exists(temp_evidence_dir)
    assert recorder.buffer_capacity >= 10
    assert not recorder._is_recording


def test_evidence_recorder_push_frame(temp_evidence_dir):
    """Test frames and telemetry are pushed to ring buffer."""
    recorder = EvidenceRecorder(
        output_dir=temp_evidence_dir,
        pre_buffer_seconds=0.5,
        post_buffer_seconds=0.5,
        fps=10.0
    )
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    telemetry = {"ear": 0.15, "state": "AWAKE"}
    
    recorder.push_frame(frame, telemetry)
    assert len(recorder._frame_buffer) == 1
    stored_frame, stored_meta = recorder._frame_buffer[0]
    assert stored_frame.shape == (100, 100, 3)
    assert stored_meta["ear"] == 0.15
    assert "timestamp" in stored_meta


def test_evidence_recorder_trigger_and_cooldown(temp_evidence_dir):
    """Test trigger initiates recording and respects cooldown."""
    recorder = EvidenceRecorder(
        output_dir=temp_evidence_dir,
        pre_buffer_seconds=0.2,
        post_buffer_seconds=0.2,
        fps=10.0,
        cooldown_seconds=1.0
    )
    
    # Trigger first recording
    triggered = recorder.trigger_recording(trigger_reason="drowsy_alert", metadata={"test": 1})
    assert triggered is True
    assert recorder._is_recording is True
    
    # Second immediate trigger should fail due to active recording / cooldown
    triggered_again = recorder.trigger_recording(trigger_reason="drowsy_alert")
    assert triggered_again is False


def test_evidence_recorder_watermark(temp_evidence_dir):
    """Test watermark banner is rendered onto frame."""
    recorder = EvidenceRecorder(output_dir=temp_evidence_dir)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    meta = {
        "fatigue_score": 85.0,
        "ear": 0.12,
        "mar": 0.35,
        "pitch": -15.0,
        "is_vehicle_moving": True,
        "is_low_light": True,
        "state": "DROWSY_ALERT"
    }
    
    watermarked = recorder._apply_evidence_watermark(frame, meta)
    assert watermarked.shape == frame.shape
    # Watermarked frame must have non-zero pixels (drawn banners)
    assert np.count_nonzero(watermarked) > 0


def test_evidence_recorder_video_creation(temp_evidence_dir):
    """Test end-to-end video recording and manifest creation."""
    saved_files = []
    
    def on_saved(filepath, manifest):
        saved_files.append((filepath, manifest))
        
    recorder = EvidenceRecorder(
        output_dir=temp_evidence_dir,
        pre_buffer_seconds=0.2,
        post_buffer_seconds=0.2,
        fps=10.0,
        cooldown_seconds=0.1,
        on_evidence_saved=on_saved
    )
    
    # Push pre-buffer frames
    for i in range(5):
        f = np.full((120, 160, 3), i * 30, dtype=np.uint8)
        recorder.push_frame(f, {"ear": 0.20, "fatigue_score": 10.0})
        
    # Trigger alert
    triggered = recorder.trigger_recording(trigger_reason="drowsy_alert", metadata={"fatigue_score": 90.0})
    assert triggered is True
    
    # Push post-buffer frames to fulfill post-frames requirement
    for i in range(12):
        f = np.full((120, 160, 3), 200, dtype=np.uint8)
        recorder.push_frame(f, {"ear": 0.08, "fatigue_score": 95.0, "state": "DROWSY_ALERT"})
        
    # Wait for async background writer thread
    time.sleep(1.0)
    
    # Check that video file and JSON manifest exist
    created_files = os.listdir(temp_evidence_dir)
    assert any(f.endswith(".mp4") or f.endswith(".avi") for f in created_files)
    assert any(f.endswith(".json") for f in created_files)
    assert len(saved_files) == 1
