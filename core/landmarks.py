"""Landmarks Module.

Provides face landmark detection using MediaPipe Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np

class LandmarkDetector:
    """Wrapper for MediaPipe FaceMesh detection."""

    def __init__(self):
        """Initialize MediaPipe Face Mesh with refined eye/lips settings."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Crucial for extra eye/iris detail points
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_landmarks(self, frame) -> np.ndarray | None:
        """Detect normalized coordinates for face landmarks.
        
        Args:
            frame: OpenCV BGR image frame.
            
        Returns:
            np.ndarray of shape (468+, 3) containing [x, y, z] coordinates,
            or None if no face was found.
        """
        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
        ])
        
        return landmarks
