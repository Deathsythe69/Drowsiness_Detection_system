"""Capture Module.

Decoupled thread for capturing frames from the camera to avoid UI lag.
"""

import cv2
import time
from PyQt6.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    """QThread to handle frame capture from a webcam."""
    frame_captured = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, camera_index: int = 0):
        """Initialize capture thread.
        
        Args:
            camera_index: Index of camera to open (e.g. 0).
        """
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self.cap = None

    def run(self):
        """Main loop that continuously pulls frames."""
        self._running = True
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                self.status_changed.emit("Connecting...")
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.status_changed.emit("Camera Disconnected. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    self.status_changed.emit("Camera Connected")

            ret, frame = self.cap.read()
            if not ret:
                self.status_changed.emit("Frame read error. Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(2)
                continue

            # Mirror the frame so it feels natural to the user
            frame = cv2.flip(frame, 1)
            self.frame_captured.emit(frame)
            time.sleep(0.03)  # Aim for ~30 FPS

    def stop(self):
        """Stop the capture thread and release the camera resource."""
        self._running = False
        self.wait()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
