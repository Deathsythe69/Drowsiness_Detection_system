"""Features Module.

Contains mathematical algorithms for:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head Pose Estimation (Pitch, Yaw, Roll) via solvePnP
- Posture slouching & nodding detection
"""

from typing import Tuple, Dict, Any, Optional
import cv2
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
    
    Args:
        landmarks: numpy array of shape (N, 3) representing landmarks.
        width: width of image frame.
        height: height of image frame.
        
    Returns:
        float: average EAR.
    """
    if landmarks is None or len(landmarks) < 468:
        return 0.0
        
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

# --- Head Pose Estimation via 3D Landmark solvePnP ---

# Generic canonical 3D facial model points (in mm)
CANONICAL_FACE_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float64)

def calculate_head_pose(
    landmarks: np.ndarray,
    width: int,
    height: int
) -> Tuple[float, float, float]:
    """Estimate 3D head rotation angles (Pitch, Yaw, Roll) using solvePnP.
    
    Args:
        landmarks: Normalized landmark array of shape (N, 3).
        width: Frame width in pixels.
        height: Frame height in pixels.
        
    Returns:
        Tuple[float, float, float]: (pitch, yaw, roll) in degrees.
            - Pitch: positive = looking up, negative = nodding/looking down
            - Yaw: positive = turning right, negative = turning left
            - Roll: positive = tilting right, negative = tilting left
    """
    if landmarks is None or len(landmarks) < 468:
        return 0.0, 0.0, 0.0
        
    # Extract 2D image points for the 6 key canonical facial anchors
    image_points = np.array([
        (landmarks[1][0] * width, landmarks[1][1] * height),       # Nose tip
        (landmarks[152][0] * width, landmarks[152][1] * height),   # Chin
        (landmarks[33][0] * width, landmarks[33][1] * height),     # Left eye outer
        (landmarks[263][0] * width, landmarks[263][1] * height),   # Right eye outer
        (landmarks[61][0] * width, landmarks[61][1] * height),     # Left mouth corner
        (landmarks[291][0] * width, landmarks[291][1] * height)    # Right mouth corner
    ], dtype=np.float64)
    
    # Camera matrix approximation based on image dimensions
    focal_length = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # Assume minimal lens distortion
    
    # Solve Perspective-n-Point
    success, rvec, _ = cv2.solvePnP(
        CANONICAL_FACE_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
        
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Decompose rotation matrix into Euler angles
    # Using RQ decomposition
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])
    
    return pitch, yaw, roll

def evaluate_posture_signal(
    pitch: float,
    yaw: float,
    roll: float,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float, str]:
    """Evaluate posture angles for head droop (nodding) and sustained distraction.
    
    Args:
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.
        roll: Roll angle in degrees.
        config: System configuration dictionary.
        
    Returns:
        Tuple: (is_slouching_or_nodding, penalty_score, event_label)
    """
    if config is None:
        config = {}
    posture_cfg = config.get("posture", {})
    if not posture_cfg.get("enabled", True):
        return False, 0.0, ""
        
    pitch_thresh = posture_cfg.get("pitch_nod_threshold", -18.0)
    yaw_thresh = posture_cfg.get("yaw_distraction_threshold", 25.0)
    penalty_weight = posture_cfg.get("slouch_penalty_weight", 1.0)
    
    # 1. Head Droop / Nodding (Pitch severely negative)
    if pitch < pitch_thresh:
        return True, penalty_weight, "head_nod_droop"
        
    # 2. Inattention / Turned Away (Yaw exceeds threshold)
    if abs(yaw) > yaw_thresh:
        return True, penalty_weight * 0.7, "distraction_turned_away"
        
    return False, 0.0, ""
