"""Landmarks Module with Multi-Person Detection and Driver ROI Isolation.

Provides multi-face landmark extraction using MediaPipe Face Mesh,
and isolates the primary subject / driver inside a specified Region of Interest.
"""

from typing import Optional, List, Dict, Any, Tuple
import cv2
import mediapipe as mp
import numpy as np

class LandmarkDetector:
    """Wrapper for MediaPipe FaceMesh detection with multi-face tracking."""

    def __init__(self, max_num_faces: int = 4):
        """Initialize MediaPipe Face Mesh.
        
        Args:
            max_num_faces: Maximum number of simultaneous faces to process.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.max_num_faces = max_num_faces
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,  # Eye and iris landmark points
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_face_info(self, face_landmarks, frame_w: int, frame_h: int) -> Dict[str, Any]:
        """Convert raw FaceMesh landmarks into structured coordinates & bounding box.
        
        Args:
            face_landmarks: MediaPipe LandmarkList.
            frame_w: Frame pixel width.
            frame_h: Frame pixel height.
            
        Returns:
            Dict containing landmarks array, bounding box (x, y, w, h), and normalized center.
        """
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
        ])
        
        # Calculate bounding box
        x_coords = landmarks[:, 0] * frame_w
        y_coords = landmarks[:, 1] * frame_h
        
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        area = box_w * box_h
        center_x = (min_x + max_x) / (2.0 * frame_w)
        center_y = (min_y + max_y) / (2.0 * frame_h)
        
        return {
            "landmarks": landmarks,
            "bbox": (min_x, min_y, box_w, box_h),
            "area": area,
            "center": (center_x, center_y),
            "is_primary": False
        }

    def detect_multi_faces(
        self,
        frame: np.ndarray,
        roi_config: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Detect all faces in the frame and isolate the primary driver.
        
        The primary driver is selected as the largest face whose center falls
        within the defined Driver Region of Interest (ROI).
        
        Args:
            frame: OpenCV BGR image frame.
            roi_config: Dict with 'x_min', 'x_max', 'y_min', 'y_max' normalized bounds.
            
        Returns:
            Tuple containing:
            - Primary driver landmarks np.ndarray (or None)
            - List of all detected face dicts (including bounding boxes & is_primary tag)
        """
        if frame is None or frame.size == 0:
            return None, []
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, []
            
        faces = [
            self.extract_face_info(fl, w, h) for fl in results.multi_face_landmarks
        ]
        
        if not faces:
            return None, []
            
        # Default ROI covers normal central driving/monitoring area
        if roi_config is None:
            roi_config = {"x_min": 0.10, "x_max": 0.90, "y_min": 0.05, "y_max": 0.95}
            
        rx_min = roi_config.get("x_min", 0.10)
        rx_max = roi_config.get("x_max", 0.90)
        ry_min = roi_config.get("y_min", 0.05)
        ry_max = roi_config.get("y_max", 0.95)
        
        # Filter faces inside the driver ROI
        candidate_indices = []
        for i, face in enumerate(faces):
            cx, cy = face["center"]
            if rx_min <= cx <= rx_max and ry_min <= cy <= ry_max:
                candidate_indices.append(i)
                
        primary_idx = None
        if candidate_indices:
            # Pick the largest face within the ROI (closest to camera)
            primary_idx = max(candidate_indices, key=lambda idx: faces[idx]["area"])
        else:
            # Fallback: pick the largest face anywhere in the frame
            primary_idx = max(range(len(faces)), key=lambda idx: faces[idx]["area"])
            
        faces[primary_idx]["is_primary"] = True
        return faces[primary_idx]["landmarks"], faces

    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Convenience method returning primary face landmarks.
        
        Args:
            frame: OpenCV BGR image frame.
            
        Returns:
            np.ndarray of shape (468+, 3) or None.
        """
        primary_landmarks, _ = self.detect_multi_faces(frame)
        return primary_landmarks
