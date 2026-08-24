"""Vehicle Motion Detection Module.

Estimates whether the vehicle is currently moving by analyzing
background optical flow in the peripheral (non-face) regions of
the video frame. Uses frame-to-frame pixel difference magnitude
as a lightweight proxy for scene motion.

This avoids false drowsiness alerts when the car is parked or
the driver is stationary at a red light / traffic jam.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple


class VehicleMotionDetector:
    """Detects vehicle motion via background frame differencing.
    
    Analyzes pixel changes in the peripheral strips of the frame
    (top, bottom, left edges) which correspond to the windshield
    view / scenery. Moving scenery = vehicle is moving.
    
    Uses a rolling average of motion scores to smooth out noise
    from lighting flickers, camera vibration, etc.
    """

    def __init__(
        self,
        motion_threshold: float = 5.0,
        window_size: int = 15,
        min_moving_ratio: float = 0.5
    ):
        """Initialize vehicle motion detector.
        
        Args:
            motion_threshold: Mean pixel difference above which a single
                frame pair is considered "moving". Typical range: 3-10.
            window_size: Number of recent frames to average over for
                the rolling motion decision.
            min_moving_ratio: Fraction of recent frames that must show
                motion for the vehicle to be considered "moving".
        """
        self.motion_threshold = motion_threshold
        self.window_size = window_size
        self.min_moving_ratio = min_moving_ratio

        self._prev_gray: Optional[np.ndarray] = None
        self._motion_history: deque = deque(maxlen=window_size)
        self._is_moving: bool = False
        self._current_motion_score: float = 0.0

    def _get_peripheral_mask(self, h: int, w: int) -> np.ndarray:
        """Create a mask covering only the peripheral scene regions.
        
        Excludes the center where faces are, and focuses on the
        top strip (sky/road ahead), left strip (side window),
        right strip (side window), and bottom strip (dashboard edge).
        
        Args:
            h: Frame height.
            w: Frame width.
            
        Returns:
            Boolean mask of shape (h, w).
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # Top strip (top 15% of frame — captures road/sky motion)
        mask[0:int(h * 0.15), :] = 255

        # Bottom strip (bottom 10% — dashboard edge / road)
        mask[int(h * 0.90):, :] = 255

        # Left strip (left 15%)
        mask[:, 0:int(w * 0.15)] = 255

        # Right strip (right 15%)
        mask[:, int(w * 0.85):] = 255

        return mask

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Analyze a new frame and update vehicle motion state.
        
        Args:
            frame: BGR video frame from the webcam.
            
        Returns:
            Tuple of (is_moving: bool, motion_score: float).
            motion_score is the mean absolute pixel difference in
            the peripheral regions (higher = more motion).
        """
        if frame is None or frame.size == 0:
            return self._is_moving, self._current_motion_score

        # Downsample to 160x120 for instant, low-latency processing (< 0.2ms)
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        h, w = gray.shape
        mask = self._get_peripheral_mask(h, w)

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            # Compute absolute frame difference
            diff = cv2.absdiff(gray, self._prev_gray)

            # Apply peripheral mask — only look at scene edges
            masked_diff = cv2.bitwise_and(diff, diff, mask=mask)

            # Mean pixel change in the masked region
            nonzero_pixels = np.count_nonzero(mask)
            if nonzero_pixels > 0:
                motion_score = float(np.sum(masked_diff)) / nonzero_pixels
            else:
                motion_score = 0.0

            self._current_motion_score = round(motion_score, 2)

            # Is this single frame "moving"?
            frame_moving = bool(motion_score > self.motion_threshold)
            self._motion_history.append(frame_moving)
        else:
            # First frame — no comparison yet
            self._motion_history.append(False)

        self._prev_gray = gray.copy()

        # Rolling decision: vehicle is moving if enough recent frames show motion
        if len(self._motion_history) > 0:
            moving_ratio = sum(self._motion_history) / len(self._motion_history)
            self._is_moving = bool(moving_ratio >= self.min_moving_ratio)
        else:
            self._is_moving = False

        return self._is_moving, self._current_motion_score

    @property
    def is_moving(self) -> bool:
        """Current vehicle motion state."""
        return self._is_moving

    @property
    def motion_score(self) -> float:
        """Latest per-frame motion score."""
        return self._current_motion_score

    def reset(self):
        """Reset motion detection state."""
        self._prev_gray = None
        self._motion_history.clear()
        self._is_moving = False
        self._current_motion_score = 0.0
