"""Unit tests for core/features.py calculations."""

import numpy as np
from core.features import calculate_distance, calculate_ear_for_eye, calculate_ear, calculate_mar

def test_calculate_distance():
    """Verify 3D coordinate distance math."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert calculate_distance(p1, p2) == 5.0

def test_calculate_ear_for_eye():
    """Verify single eye aspect ratio calculation logic."""
    # Let p1=(0,0), p4=(10,0) (horizontal distance = 10)
    # Let p2=(3,2), p6=(3,-2) (vertical distance = 4)
    # Let p3=(7,2), p5=(7,-2) (vertical distance = 4)
    # EAR = (4 + 4) / (2 * 10) = 8 / 20 = 0.4
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 2.0, 0.0])
    p3 = np.array([7.0, 2.0, 0.0])
    p4 = np.array([10.0, 0.0, 0.0])
    p5 = np.array([7.0, -2.0, 0.0])
    p6 = np.array([3.0, -2.0, 0.0])
    
    ear = calculate_ear_for_eye(p1, p2, p3, p4, p5, p6)
    assert abs(ear - 0.4) < 1e-6

def test_calculate_ear_none():
    """Verify calculate_ear returns 0 when landmarks are missing."""
    assert calculate_ear(None, 640, 480) == 0.0
    assert calculate_ear(np.zeros((10, 3)), 640, 480) == 0.0

def test_calculate_mar_none():
    """Verify calculate_mar returns 0 when landmarks are missing."""
    assert calculate_mar(None, 640, 480) == 0.0
    assert calculate_mar(np.zeros((10, 3)), 640, 480) == 0.0
