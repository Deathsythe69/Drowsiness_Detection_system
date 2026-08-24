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

    def _open_camera(self) -> cv2.VideoCapture:
        """Attempt to open camera using DirectShow on Windows, with fallback."""
        import sys
        cap = None
        # On Windows, DirectShow (CAP_DSHOW) avoids MSMF sample reader failure
        if sys.platform == "win32":
            try:
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
            except Exception:
                pass

        # Fallback to default backend
        cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        """Main loop that continuously pulls frames."""
        self._running = True
        consecutive_read_failures = 0

        while self._running:
            if self.cap is None or not self.cap.isOpened():
                self.status_changed.emit("Connecting camera...")
                self.cap = self._open_camera()
                if self.cap is None or not self.cap.isOpened():
                    self.status_changed.emit("Camera Disconnected. Retrying...")
                    time.sleep(1.5)
                    continue
                else:
                    self.status_changed.emit("Camera Connected")
                    consecutive_read_failures = 0

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 5:
                    self.status_changed.emit("Frame read error. Reconnecting...")
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = None
                    consecutive_read_failures = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.05)
                continue

            consecutive_read_failures = 0

            # Mirror the frame so it feels natural to the user
            frame = cv2.flip(frame, 1)
            self.frame_captured.emit(frame)
            time.sleep(0.005)  # Minimal sleep to yield GIL without dropping FPS

    def stop(self):
        """Stop the capture thread and release the camera resource."""
        self._running = False
        self.wait()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
