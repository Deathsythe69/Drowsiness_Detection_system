"""Unit tests for multi-face detection and driver ROI isolation."""

import numpy as np
from core.landmarks import LandmarkDetector

def test_landmark_detector_initialization():
    """Verify LandmarkDetector instantiates with multi-face support."""
    detector = LandmarkDetector(max_num_faces=4)
    assert detector.max_num_faces == 4
    assert detector.face_mesh is not None

def test_extract_face_info():
    """Verify bounding box and center extraction."""
    detector = LandmarkDetector(max_num_faces=2)
    
    # Mock face landmarks object with 468 fake points
    class MockPoint:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            
    class MockLandmarkList:
        def __init__(self):
            # Points between x: [0.2, 0.4] and y: [0.3, 0.6]
            self.landmark = [MockPoint(0.2, 0.3, 0.0), MockPoint(0.4, 0.6, 0.0)]
            
    face_info = detector.extract_face_info(MockLandmarkList(), 640, 480)
    assert face_info["bbox"] == (128, 144, 128, 144)
    assert face_info["area"] == 128 * 144
    assert 0.29 <= face_info["center"][0] <= 0.31
    assert 0.44 <= face_info["center"][1] <= 0.46
