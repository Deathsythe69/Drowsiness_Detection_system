"""Features Module.

Contains math routines for calculating EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).
"""

import numpy as np

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: First coordinate (x, y, z).
        p2: Second coordinate (x, y, z).
        
    Returns:
        float: Euclidean distance.
    """
    return float(np.linalg.norm(p1 - p2))

def calculate_ear_for_eye(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
    p4: np.ndarray, p5: np.ndarray, p6: np.ndarray
) -> float:
    """Calculate Eye Aspect Ratio (EAR) for a single eye.
    
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)
    horizontal = calculate_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_ear(landmarks: np.ndarray, width: int, height: int) -> float:
    """Calculate average EAR across left and right eyes.
    
    Coordinates are scaled to pixel values using frame dimensions.
    
    Args:
        landmarks: numpy array of shape (N, 3) representing landmarks.
        width: width of image frame.
        height: height of image frame.
        
    Returns:
        float: average EAR.
    """
    if landmarks is None or len(landmarks) < 468:
        return 0.0
        
    # Scale coordinate components to pixels
    pixel_landmarks = landmarks * np.array([width, height, 1])
    
    # Left eye landmarks:
    # p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    left_ear = calculate_ear_for_eye(
        pixel_landmarks[362], pixel_landmarks[385], pixel_landmarks[387],
        pixel_landmarks[263], pixel_landmarks[373], pixel_landmarks[380]
    )
    
    # Right eye landmarks:
    # p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    right_ear = calculate_ear_for_eye(
        pixel_landmarks[33], pixel_landmarks[160], pixel_landmarks[158],
        pixel_landmarks[133], pixel_landmarks[153], pixel_landmarks[144]
    )
    
    return float((left_ear + right_ear) / 2.0)

def calculate_mar(landmarks: np.ndarray, width: int, height: int) -> float:
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection.
    
    Formula: MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
    
    Args:
        landmarks: numpy array of shape (N, 3) representing landmarks.
        width: width of image frame.
        height: height of image frame.
        
    Returns:
        float: MAR value.
    """
    if landmarks is None or len(landmarks) < 318:
        return 0.0
        
    pixel_landmarks = landmarks * np.array([width, height, 1])
    
    # Inner lip points:
    # p1=78, p2=81, p3=13, p4=311, p5=308, p6=317, p7=14, p8=87
    p1 = pixel_landmarks[78]
    p2 = pixel_landmarks[81]
    p3 = pixel_landmarks[13]
    p4 = pixel_landmarks[311]
    p5 = pixel_landmarks[308]
    p6 = pixel_landmarks[317]
    p7 = pixel_landmarks[14]
    p8 = pixel_landmarks[87]
    
    vertical_1 = calculate_distance(p2, p8)
    vertical_2 = calculate_distance(p3, p7)
    vertical_3 = calculate_distance(p4, p6)
    horizontal = calculate_distance(p1, p5)
    
    if horizontal == 0:
        return 0.0
        
    return float((vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal))
