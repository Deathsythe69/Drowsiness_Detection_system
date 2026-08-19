"""Main Monitor Window.

Coordinates the camera thread, MediaPipe landmark detector, FSM classifier,
sound alerts, SQLite logging, and dashboard user interface.
"""

import time
import cv2
import numpy as np
from PyQt6.QtWidgets import (
     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,   QProgressBar, QMessageBox )

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QImage, QPixmap

from core.capture import CaptureThread
from core.landmarks import LandmarkDetector
from core.features import calculate_ear, calculate_mar
from core.classifier import StateClassifier, State
from core.alerts import AlertManager
from storage.models import Session, Event
from storage.db import init_db
from ui.settings_window import SettingsWindow
from ui.summary_window import SummaryWindow
from ui.math_puzzle_window import MathPuzzleDialog

class MainWindow(QMainWindow):
    """Primary GUI Application Window for Attention Monitoring."""

    def __init__(self, config: dict):
        """Initialize main window.
        
        Args:
            config: Loaded config dictionary.
        """
        super().__init__()
        self.config = config
        self.setWindowTitle("Attention & Drowsiness Monitoring System")
        self.resize(1000, 650)
        self.setObjectName("main_window")

        # Initialize engines
        init_db()  # Ensure SQLite tables exist
        self.detector = LandmarkDetector()
        self.classifier = StateClassifier(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Thread handles
        self.capture_thread = None
        
        # Active session state tracking
        self.active_session = None
        self.session_seconds = 0
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self.tick_session_timer)

        # Baseline calibration variables
        self.calibration_active = False
        self.calibration_frames = []
        self.calibration_countdown = 10
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.tick_calibration)

        # Database logging throttle: log normal states every 1 second (approx 30 frames)
        self.log_throttle_counter = 0

        self.init_ui()
        self.start_camera()

    def init_ui(self):
        """Build the dashboard visual layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- LEFT PANEL: Webcam display ---
        left_layout = QVBoxLayout()
        
        self.camera_feed_lbl = QLabel("Camera Offline")
        self.camera_feed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed_lbl.setFrameStyle(QFrame.Shape.StyledPanel)
        self.camera_feed_lbl.setMinimumSize(640, 480)
        self.camera_feed_lbl.setStyleSheet(
            "background-color: #1a1a1e; border: 2px solid #2e2e33; border-radius: 12px; font-weight: bold; color: #71717a;"
        )
        left_layout.addWidget(self.camera_feed_lbl)

        # Bottom buttons row
        btn_row = QHBoxLayout()
        
        self.camera_toggle_btn = QPushButton("Stop Monitoring")
        self.camera_toggle_btn.clicked.connect(self.toggle_monitoring)
        btn_row.addWidget(self.camera_toggle_btn)
        
        self.calibrate_btn = QPushButton("Quick Calibrate")
        self.calibrate_btn.setObjectName("secondary_btn")
        self.calibrate_btn.clicked.connect(self.trigger_calibration)
        btn_row.addWidget(self.calibrate_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("secondary_btn")
        self.settings_btn.clicked.connect(self.open_settings)
        btn_row.addWidget(self.settings_btn)

        self.summary_btn = QPushButton("History Summary")
        self.summary_btn.setObjectName("secondary_btn")
        self.summary_btn.clicked.connect(self.open_summary)
        btn_row.addWidget(self.summary_btn)

        left_layout.addLayout(btn_row)
        main_layout.addLayout(left_layout, stretch=2)

        # --- RIGHT PANEL: Status readout cards ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # 1. Status Indicator Card
        self.status_card = QFrame()
        self.status_card.setObjectName("status_card")
        self.status_card.setFrameStyle(QFrame.Shape.StyledPanel)
        self.status_card.setStyleSheet("background-color: #27272a; border-radius: 12px;")
        
        sc_layout = QVBoxLayout(self.status_card)
        sc_layout.addWidget(QLabel("ATTENTION STATE"), alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_status = QLabel("MONITOR INACTIVE")
        self.lbl_status.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        sc_layout.addWidget(self.lbl_status, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.status_card)

        # Calibration Banner
        self.calibration_banner = QLabel("")
        self.calibration_banner.setWordWrap(True)
        self.calibration_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_banner.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.calibration_banner.hide()
        right_layout.addWidget(self.calibration_banner)

        # 2. Session Metrics Card
        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_card.setFrameStyle(QFrame.Shape.StyledPanel)
        mc_layout = QVBoxLayout(metrics_card)
        
        self.lbl_timer = QLabel("Session Duration: 00:00")
        self.lbl_timer.setStyleSheet("font-size: 16px; font-weight: bold;")
        mc_layout.addWidget(self.lbl_timer)

        # EAR Progress
        mc_layout.addWidget(QLabel("Eye Opening Ratio (EAR):"))
        self.ear_bar = QProgressBar()
        self.ear_bar.setRange(0, 100)
        self.ear_bar.setValue(0)
        self.ear_bar.setFormat("%.2f")
        mc_layout.addWidget(self.ear_bar)
        self.lbl_ear_threshold = QLabel("Alert Threshold: 0.21")
        self.lbl_ear_threshold.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        mc_layout.addWidget(self.lbl_ear_threshold)

        # MAR Progress
        mc_layout.addWidget(QLabel("Mouth Opening Ratio (MAR):"))
        self.mar_bar = QProgressBar()
        self.mar_bar.setRange(0, 100)
        self.mar_bar.setValue(0)
        self.mar_bar.setFormat("%.2f")
        mc_layout.addWidget(self.mar_bar)
        self.lbl_mar_threshold = QLabel("Yawn Threshold: 0.60")
        self.lbl_mar_threshold.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        mc_layout.addWidget(self.lbl_mar_threshold)

        # Fatigue Score Progress
        mc_layout.addWidget(QLabel("Cumulative Fatigue Score:"))
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(0)
        self.fatigue_bar.setTextVisible(True)
        mc_layout.addWidget(self.fatigue_bar)

        right_layout.addWidget(metrics_card)

        # 3. Dismiss Alarm button
        self.dismiss_btn = QPushButton("DISMISS AUDIO ALERT")
        self.dismiss_btn.setObjectName("danger_btn")
        self.dismiss_btn.setEnabled(False)
        self.dismiss_btn.clicked.connect(self.dismiss_alert)
        right_layout.addWidget(self.dismiss_btn)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

        self.update_threshold_labels()

    def update_threshold_labels(self):
        """Update textual threshold readouts."""
        ear_thr = self.config.get("thresholds", {}).get("ear_threshold", 0.21)
        mar_thr = self.config.get("thresholds", {}).get("mar_threshold", 0.60)
        self.lbl_ear_threshold.setText(f"Alert Threshold: {ear_thr:.2f}")
        self.lbl_mar_threshold.setText(f"Yawn Threshold: {mar_thr:.2f}")

    def start_camera(self):
        """Initialize and start capture thread."""
        camera_idx = self.config.get("camera", {}).get("index", 0)
        
        self.capture_thread = CaptureThread(camera_idx)
        self.capture_thread.frame_captured.connect(self.on_frame_captured)
        self.capture_thread.status_changed.connect(self.on_status_changed)
        self.capture_thread.start()

        # Start tracking Session in database
        self.active_session = Session.create(
            user_label="Self-Monitoring",
            baseline_ear=self.config.get("thresholds", {}).get("ear_threshold", 0.21)
        )
        self.session_seconds = 0
        self.session_timer.start(1000)

    def stop_camera(self):
        """Stop capture thread and wrap up session."""
        self.session_timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
            
        if self.active_session is not None:
            self.active_session.end()
            self.active_session = None

        self.camera_feed_lbl.setText("Camera Offline")
        self.camera_feed_lbl.setStyleSheet(
            "background-color: #1a1a1e; border: 2px solid #2e2e33; border-radius: 12px; font-weight: bold; color: #71717a;"
        )
        self.lbl_status.setText("MONITOR INACTIVE")
        self.status_card.setStyleSheet("background-color: #27272a; border-radius: 12px;")
        self.alert_manager.stop_alarm()
        self.dismiss_btn.setEnabled(False)

    def toggle_monitoring(self):
        """Start/Stop webcam monitoring handle."""
        if self.capture_thread is not None:
            self.stop_camera()
            self.camera_toggle_btn.setText("Start Monitoring")
        else:
            self.start_camera()
            self.camera_toggle_btn.setText("Stop Monitoring")

    @pyqtSlot(object)
    def on_frame_captured(self, frame):
        """Process incoming video frames, compute metrics, and draw UI update."""
        h, w, _ = frame.shape
        landmarks = self.detector.detect_landmarks(frame)
        has_face = landmarks is not None

        # Features calculation
        ear = 0.0
        mar = 0.0
        if has_face:
            ear = calculate_ear(landmarks, w, h)
            mar = calculate_mar(landmarks, w, h)
            
            # If calibration scan is actively running
            if self.calibration_active:
                self.calibration_frames.append(ear)

        # FSM Classifier Evaluation
        state, score, events = self.classifier.process_frame(has_face, ear, mar)
        
        # UI Updates
        self.update_status_display(state, score)
        self.update_metrics_bars(ear, mar, score)

        # Alarm loop check
        if state == State.DROWSY_ALERT:
            self.alert_manager.play_alarm()
            self.dismiss_btn.setEnabled(True)
        else:
            # Shut down sound if state is fixed and not manually muted
            if self.alert_manager.is_playing and not self.alert_manager.sound_enabled:
                pass  # user pressed dismiss, let them stay quiet
            elif state != State.DROWSY_ALERT:
                self.alert_manager.stop_alarm()
                # reset sound enabled flag if they are awake again
                self.alert_manager.sound_enabled = True
                self.dismiss_btn.setEnabled(False)

        # Database Events Logging
        if self.active_session is not None:
            self.log_throttle_counter += 1
            
            # Log any explicit events immediately
            for evt in events:
                Event.log(self.active_session.id, evt, ear, mar)
                
            # Log raw state values every 30 frames (~1 second) to construct smooth charts
            if self.log_throttle_counter >= 30:
                self.log_throttle_counter = 0
                state_str = state.value.lower()
                Event.log(self.active_session.id, state_str, ear, mar)

        # Render facial landmark overlays if preference enabled
        if has_face and self.config.get("ui", {}).get("show_landmarks", True):
            self.draw_landmarks(frame, landmarks, w, h)

        # Display Frame in QLabel
        self.display_image(frame)

    def draw_landmarks(self, frame, landmarks, w, h):
        """Draw colored mesh points on the OpenCV frame."""
        # Key landmark points mappings
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        inner_lips = [78, 81, 13, 311, 308, 317, 14, 87]
        accent_points = [168, 6, 152]  # nose bridge, nose tip, chin
        
        # Eyes -> Cyan
        for idx in left_eye + right_eye:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            
        # Lips -> Magenta
        for idx in inner_lips:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

        # Accents -> White
        for idx in accent_points:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (230, 230, 230), -1)

    def display_image(self, frame):
        """Convert BGR cv2 image to QImage and draw to QLabel."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.camera_feed_lbl.width(), 
            self.camera_feed_lbl.height(), 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.camera_feed_lbl.setPixmap(pixmap)

    def update_status_display(self, state: State, score: float):
        """Update status label and box colors."""
        if state == State.AWAKE:
            self.lbl_status.setText("AWAKE")
            self.status_card.setStyleSheet("background-color: #059669; border-radius: 12px;") # Emerald Green
        elif state == State.DROWSY_WARNING:
            self.lbl_status.setText("DROWSY WARNING")
            self.status_card.setStyleSheet("background-color: #d97706; border-radius: 12px;") # Amber Yellow
        elif state == State.DROWSY_ALERT:
            self.lbl_status.setText("DROWSY ALERT!")
            self.status_card.setStyleSheet("background-color: #dc2626; border-radius: 12px;") # Rose Red
        elif state == State.NO_FACE:
            self.lbl_status.setText("NO FACE DETECTED")
            self.status_card.setStyleSheet("background-color: #4b5563; border-radius: 12px;") # Gray

    def update_metrics_bars(self, ear: float, mar: float, score: float):
        """Map numerical metrics to progress bars (scale 0-100)."""
        # Map EAR typically from 0.0 -> 0.40 into 0 -> 100
        ear_percent = int(min(100, max(0, (ear / 0.40) * 100)))
        self.ear_bar.setValue(ear_percent)
        self.ear_bar.setFormat(f"{ear:.3f}")

        # Map MAR typically from 0.0 -> 0.80 into 0 -> 100
        mar_percent = int(min(100, max(0, (mar / 0.80) * 100)))
        self.mar_bar.setValue(mar_percent)
        self.mar_bar.setFormat(f"{mar:.3f}")

        # Fatigue score direct
        self.fatigue_bar.setValue(int(score))

    def dismiss_alert(self):
        """Temporarily mute active sound loop alert after verifying with a math puzzle."""
        from PyQt6.QtWidgets import QDialog
        dialog = MathPuzzleDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # User successfully solved the math problem!
            self.alert_manager.stop_alarm()
            self.alert_manager.sound_enabled = True  # Re-enable/re-arm sound
            self.dismiss_btn.setEnabled(False)
            
            # Reset the classifier to clean the state and fatigue score
            self.classifier.reset()
            
            # Force status display to AWAKE
            self.update_status_display(State.AWAKE, 0.0)
            self.update_metrics_bars(0.35, 0.15, 0.0) # Reset bars to safe awake values

    @pyqtSlot(str)
    def on_status_changed(self, status_msg):
        """Handle webcam status connection announcements."""
        # Only overwrite camera display text when thread is starting/disconnected
        if "Disconnected" in status_msg or "Connecting" in status_msg:
            self.camera_feed_lbl.setText(status_msg)

    def tick_session_timer(self):
        """Increment session timer display."""
        self.session_seconds += 1
        m, s = divmod(self.session_seconds, 60)
        self.lbl_timer.setText(f"Session Duration: {m:02d}:{s:02d}")

    # --- Calibration Routine ---
    def trigger_calibration(self):
        """Begin 10-second calibration routine."""
        if self.capture_thread is None:
            QMessageBox.warning(self, "Camera Offline", "Start camera before calibrating.")
            return

        self.calibration_active = True
        self.calibration_frames = []
        self.calibration_countdown = 10
        
        self.calibrate_btn.setEnabled(False)
        self.calibration_banner.show()
        self.calibration_banner.setText(f"Calibrating baseline... Please look straight at camera.\nCountdown: {self.calibration_countdown}s")
        
        self.calibration_timer.start(1000)

    def tick_calibration(self):
        """Tick down calibration timer."""
        self.calibration_countdown -= 1
        if self.calibration_countdown > 0:
            self.calibration_banner.setText(f"Calibrating baseline... Please look straight at camera.\nCountdown: {self.calibration_countdown}s")
        else:
            # End calibration
            self.calibration_timer.stop()
            self.calibration_active = False
            self.calibrate_btn.setEnabled(True)
            self.calibration_banner.hide()
            
            # Compute results
            if len(self.calibration_frames) > 50:
                mean_ear = float(np.mean(self.calibration_frames))
                # Sane adjustments: Alert threshold at 70% of baseline EAR
                new_threshold = round(mean_ear * 0.70, 3)
                
                # Write to config
                self.config["thresholds"]["ear_threshold"] = new_threshold
                
                # Propagate to components
                self.classifier.update_config(self.config)
                self.update_threshold_labels()
                
                # Also save baseline in db if session is open
                if self.active_session is not None:
                    self.active_session.baseline_ear = mean_ear
                
                QMessageBox.information(
                    self, "Calibration Completed",
                    f"Baseline EAR: {mean_ear:.3f}\nNew Alert Threshold (70%): {new_threshold:.3f}"
                )
            else:
                QMessageBox.warning(self, "Calibration Failed", "Not enough face frames captured. Try again.")

    # --- Menu navigation ---
    def open_settings(self):
        """Open settings dialog."""
        # If camera is active, settings changes may require reloading index
        dlg = SettingsWindow(self.config, self)
        # Connect calibration click on settings panel to trigger quick calibration in monitor
        dlg.calibration_requested.connect(self.trigger_calibration)
        
        # Connect settings saves
        def on_saved(new_config):
            self.config = new_config
            self.classifier.update_config(self.config)
            self.alert_manager.update_config(self.config)
            self.update_threshold_labels()
            # If camera index changed, reconnect
            camera_idx = self.config.get("camera", {}).get("index", 0)
            if self.capture_thread is not None and self.capture_thread.camera_index != camera_idx:
                self.stop_camera()
                self.start_camera()

        dlg.config_saved.connect(on_saved)
        dlg.exec()

    def open_summary(self):
        """Open history summary dashboard."""
        dlg = SummaryWindow(self)
        dlg.exec()

    def closeEvent(self, event):
        """Clean up threads when closing the main window."""
        self.stop_camera()
        event.accept()

