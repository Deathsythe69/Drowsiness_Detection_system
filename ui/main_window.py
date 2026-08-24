"""Main Monitor Window.

Coordinates:
- Webcam capture thread & disconnect resilience
- Adaptive low-light preprocessing & CLAHE enhancement
- MediaPipe multi-face detection & primary driver ROI isolation
- EAR, MAR, and solvePnP Head Pose (Pitch, Yaw, Roll) estimation
- FSM Classifier with rolling yawn frequency escalation & posture penalties
- Device battery health monitoring
- User calibration profiles & Admin PIN alarm override
- SQLite session & event logging
"""

import time
import yaml
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

from core.capture import CaptureThread
from core.landmarks import LandmarkDetector
from core.features import calculate_ear, calculate_mar, calculate_head_pose
from core.preprocessing import preprocess_frame
from core.classifier import StateClassifier, State
from core.alerts import AlertManager
from core.system_health import get_battery_status
from core.motion_detector import VehicleMotionDetector
from core.shared_state import SharedState, BuzzerCommand
from core.remote_admin import start_remote_admin_server
from storage.models import Session, Event, UserProfile
from storage.db import init_db
from ui.settings_window import SettingsWindow
from ui.summary_window import SummaryWindow
from ui.math_puzzle_window import MathPuzzleDialog
from ui.admin_dialog import AdminOverrideDialog
from ui.profile_dialog import ProfileManagerDialog
from ui.qr_dialog import WiFiAccessDialog

class MainWindow(QMainWindow):
    """Primary GUI Application Window for Attention & Fatigue Monitoring."""

    def __init__(self, config: dict):
        """Initialize main window.
        
        Args:
            config: Loaded config dictionary.
        """
        super().__init__()
        self.config = config
        self.setWindowTitle("AI Attention & Drowsiness Monitoring System")
        self.resize(1080, 700)
        self.setObjectName("main_window")

        # Initialize engines
        init_db()
        self.detector = LandmarkDetector(max_num_faces=self.config.get("multi_face", {}).get("max_faces", 4))
        self.classifier = StateClassifier(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Vehicle motion detector (driving vs stationary/parked)
        motion_cfg = self.config.get("motion", {})
        self.motion_detector = VehicleMotionDetector(
            motion_threshold=motion_cfg.get("motion_threshold", 4.0),
            window_size=motion_cfg.get("window_size", 15),
            min_moving_ratio=motion_cfg.get("min_moving_ratio", 0.4)
        )
        self.simulate_driving = motion_cfg.get("simulate_driving", False)
        self.is_vehicle_moving = self.simulate_driving
        self.current_motion_score = 99.0 if self.simulate_driving else 0.0
        
        # Thread & session handles
        self.capture_thread = None
        self.active_session = None
        self.session_seconds = 0
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self.tick_session_timer)

        # Periodic battery monitoring timer (every 5 seconds)
        self.battery_timer = QTimer(self)
        self.battery_timer.timeout.connect(self.check_battery_health)

        # Baseline calibration variables
        self.calibration_active = False
        self.calibration_frames = []
        self.calibration_countdown = 10
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.tick_calibration)

        # Logging throttle (every ~30 frames ≈ 1s for smooth charts)
        self.log_throttle_counter = 0

        # Latest runtime diagnostics
        self.is_low_light_state = False
        self.current_luminance = 100.0

        # Shared state bridge for remote admin
        self.shared_state = SharedState()
        self.remote_admin_url = ""

        self.init_ui()
        self.start_camera()
        self.check_battery_health()
        self.battery_timer.start(5000)

        # Start remote admin web panel if enabled
        admin_cfg = self.config.get("admin", {})
        if admin_cfg.get("remote_enabled", True):
            admin_pin = str(admin_cfg.get("pin", "1234"))
            admin_port = admin_cfg.get("remote_port", 8080)
            try:
                self.remote_admin_url = start_remote_admin_server(admin_pin, admin_port)
                self.remote_url_label.setText(f"Remote Admin: {self.remote_admin_url}")
                self.remote_url_label.show()
            except Exception as e:
                print(f"Failed to start remote admin server: {e}")
                self.remote_url_label.setText("Remote Admin: Failed to start")
                self.remote_url_label.show()

    def init_ui(self):
        """Build dashboard layout and HUD components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- LEFT PANEL: Video canvas & control toolbar ---
        left_layout = QVBoxLayout()
        
        # Video feed label
        self.camera_feed_lbl = QLabel("Initializing Video Feed...")
        self.camera_feed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed_lbl.setFrameStyle(QFrame.Shape.StyledPanel)
        self.camera_feed_lbl.setMinimumSize(640, 480)
        self.camera_feed_lbl.setStyleSheet(
            "background-color: #1a1a1e; border: 2px solid #2e2e33; border-radius: 12px; font-weight: bold; color: #71717a;"
        )
        left_layout.addWidget(self.camera_feed_lbl)

        # System health banner (Low light / Battery / Camera)
        self.health_banner = QLabel("")
        self.health_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.health_banner.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 12px;")
        self.health_banner.hide()
        left_layout.addWidget(self.health_banner)

        # Remote admin URL display
        self.remote_url_label = QLabel("")
        self.remote_url_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.remote_url_label.setStyleSheet(
            "color: #60a5fa; font-weight: bold; font-size: 12px; "
            "background-color: rgba(37, 99, 235, 0.1); border: 1px solid #2563eb; "
            "border-radius: 6px; padding: 4px 8px;"
        )
        self.remote_url_label.hide()
        left_layout.addWidget(self.remote_url_label)

        # Bottom buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        self.camera_toggle_btn = QPushButton("Stop Monitoring")
        self.camera_toggle_btn.clicked.connect(self.toggle_monitoring)
        btn_row.addWidget(self.camera_toggle_btn)
        
        self.calibrate_btn = QPushButton("Quick Calibrate")
        self.calibrate_btn.setObjectName("secondary_btn")
        self.calibrate_btn.clicked.connect(self.trigger_calibration)
        btn_row.addWidget(self.calibrate_btn)

        self.profiles_btn = QPushButton("Profiles")
        self.profiles_btn.setObjectName("secondary_btn")
        self.profiles_btn.clicked.connect(self.open_profiles)
        btn_row.addWidget(self.profiles_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("secondary_btn")
        self.settings_btn.clicked.connect(self.open_settings)
        btn_row.addWidget(self.settings_btn)

        self.summary_btn = QPushButton("Analytics Summary")
        self.summary_btn.setObjectName("secondary_btn")
        self.summary_btn.clicked.connect(self.open_summary)
        btn_row.addWidget(self.summary_btn)

        self.wifi_btn = QPushButton("📱 Wi-Fi / QR")
        self.wifi_btn.setObjectName("secondary_btn")
        self.wifi_btn.setStyleSheet("background-color: #1e3a8a; color: #93c5fd; font-weight: bold;")
        self.wifi_btn.setToolTip("Open scannable QR code for phone/tablet access over Wi-Fi or Hotspot")
        self.wifi_btn.clicked.connect(self.open_wifi_access)
        btn_row.addWidget(self.wifi_btn)

        left_layout.addLayout(btn_row)
        main_layout.addLayout(left_layout, stretch=2)

        # --- RIGHT PANEL: Telemetry cards & alert controls ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # 1. State Indicator Card
        self.status_card = QFrame()
        self.status_card.setObjectName("status_card")
        self.status_card.setFrameStyle(QFrame.Shape.StyledPanel)
        self.status_card.setStyleSheet("background-color: #27272a; border-radius: 12px;")
        
        sc_layout = QVBoxLayout(self.status_card)
        sc_layout.addWidget(QLabel("ATTENTION STATE"), alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_status = QLabel("INITIALIZING...")
        self.lbl_status.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff;")
        sc_layout.addWidget(self.lbl_status, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.status_card)

        # Calibration status banner
        self.calibration_banner = QLabel("")
        self.calibration_banner.setWordWrap(True)
        self.calibration_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_banner.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.calibration_banner.hide()
        right_layout.addWidget(self.calibration_banner)

        # 2. Real-Time Telemetry Card
        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_card.setFrameStyle(QFrame.Shape.StyledPanel)
        mc_layout = QVBoxLayout(metrics_card)
        mc_layout.setSpacing(6)
        
        # Session duration & battery readout
        header_row = QHBoxLayout()
        self.lbl_timer = QLabel("Duration: 00:00")
        self.lbl_timer.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_row.addWidget(self.lbl_timer)

        self.lbl_battery = QLabel("Battery: --%")
        self.lbl_battery.setStyleSheet("font-size: 12px; color: #10b981; font-weight: bold;")
        header_row.addWidget(self.lbl_battery, alignment=Qt.AlignmentFlag.AlignRight)
        mc_layout.addLayout(header_row)

        # Eye Opening Ratio (EAR)
        mc_layout.addWidget(QLabel("Eye Opening Ratio (EAR):"))
        self.ear_bar = QProgressBar()
        self.ear_bar.setRange(0, 100)
        self.ear_bar.setValue(0)
        self.ear_bar.setFormat("%.3f")
        mc_layout.addWidget(self.ear_bar)
        self.lbl_ear_threshold = QLabel("Alert Threshold: 0.21")
        self.lbl_ear_threshold.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        mc_layout.addWidget(self.lbl_ear_threshold)

        # Mouth Opening Ratio (MAR)
        mc_layout.addWidget(QLabel("Mouth Opening Ratio (MAR):"))
        self.mar_bar = QProgressBar()
        self.mar_bar.setRange(0, 100)
        self.mar_bar.setValue(0)
        self.mar_bar.setFormat("%.3f")
        mc_layout.addWidget(self.mar_bar)
        self.lbl_mar_threshold = QLabel("Yawn Threshold: 0.60")
        self.lbl_mar_threshold.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        mc_layout.addWidget(self.lbl_mar_threshold)

        # Rolling Yawn Frequency Counter
        self.lbl_yawn_freq = QLabel("Rolling 5m Yawns: 0 / 3")
        self.lbl_yawn_freq.setStyleSheet("font-size: 12px; color: #e4e4e7; font-weight: bold;")
        mc_layout.addWidget(self.lbl_yawn_freq)

        # Head Pose Pitch / Yaw
        self.lbl_head_pose = QLabel("Head Pose: Pitch 0.0° | Yaw 0.0°")
        self.lbl_head_pose.setStyleSheet("font-size: 11px; color: #60a5fa;")
        mc_layout.addWidget(self.lbl_head_pose)

        # Vehicle Motion State & Simulated Driving Test Switch
        motion_card = QFrame()
        motion_card.setStyleSheet("background: rgba(39, 39, 42, 0.6); border: 1px solid #3f3f46; border-radius: 8px;")
        motion_box = QVBoxLayout(motion_card)
        motion_box.setContentsMargins(8, 6, 8, 6)
        motion_box.setSpacing(6)

        self.lbl_motion = QLabel("Vehicle State: Stationary / Parked")
        self.lbl_motion.setStyleSheet("font-size: 12px; color: #f59e0b; font-weight: bold; border: none;")
        motion_box.addWidget(self.lbl_motion)

        self.drive_sim_btn = QPushButton("🚗 Test Driving Mode: OFF")
        self.drive_sim_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drive_sim_btn.setToolTip("Click to toggle simulated driving mode on/off (enables drowsiness alarm testing at your desk)")
        self.drive_sim_btn.clicked.connect(self.toggle_simulated_driving)
        motion_box.addWidget(self.drive_sim_btn)

        mc_layout.addWidget(motion_card)

        # Fatigue Score
        mc_layout.addWidget(QLabel("Cumulative Fatigue Score:"))
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(0)
        mc_layout.addWidget(self.fatigue_bar)

        right_layout.addWidget(metrics_card)

        # 3. Action Buttons (User Dismiss vs Admin Override)
        action_box = QVBoxLayout()
        action_box.setSpacing(6)

        self.dismiss_btn = QPushButton("DISMISS ALARM (MATH PUZZLE)")
        self.dismiss_btn.setObjectName("danger_btn")
        self.dismiss_btn.setEnabled(False)
        self.dismiss_btn.clicked.connect(self.dismiss_alert)
        action_box.addWidget(self.dismiss_btn)

        self.admin_btn = QPushButton("ADMIN PIN SILENCE OVERRIDE")
        self.admin_btn.setStyleSheet("background-color: #b45309; color: white; font-weight: bold;")
        self.admin_btn.setEnabled(False)
        self.admin_btn.clicked.connect(self.admin_override)
        action_box.addWidget(self.admin_btn)

        right_layout.addLayout(action_box)
        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

        self.update_threshold_labels()
        self.update_simulated_driving_button()

    def update_simulated_driving_button(self):
        """Update the visual styling and text of the simulated driving mode toggle switch."""
        if self.simulate_driving:
            self.drive_sim_btn.setText("🚗 Test Driving Mode: ON (Active)")
            self.drive_sim_btn.setStyleSheet(
                "background-color: #059669; color: #ffffff; font-weight: bold; "
                "font-size: 11px; border: 1px solid #10b981; border-radius: 6px; padding: 6px;"
            )
            self.drive_sim_btn.setToolTip("Test Driving Mode is ON: Motion gating is bypassed so drowsiness alarms sound at desk.")
        else:
            self.drive_sim_btn.setText("🚗 Test Driving Mode: OFF")
            self.drive_sim_btn.setStyleSheet(
                "background-color: #27272a; color: #a1a1aa; font-weight: 500; "
                "font-size: 11px; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px;"
            )
            self.drive_sim_btn.setToolTip("Test Driving Mode is OFF: Real optical flow background motion detection active.")

    def toggle_simulated_driving(self):
        """Toggle simulated driving mode for desk testing without vehicle motion."""
        self.simulate_driving = not self.simulate_driving
        self.config.setdefault("motion", {})["simulate_driving"] = self.simulate_driving
        self.update_simulated_driving_button()
        
        # Save to config.yaml so choice persists
        try:
            with open("config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error persisting simulated driving setting to config.yaml: {e}")

        # Update HUD motion label immediately
        if self.simulate_driving:
            self.is_vehicle_moving = True
            self.current_motion_score = 99.0
            self.lbl_motion.setText("Vehicle: 🚗 DRIVING (TEST SIMULATION)")
            self.lbl_motion.setStyleSheet("font-size: 12px; color: #10b981; font-weight: bold; border: none;")
        else:
            self.is_vehicle_moving = False
            self.current_motion_score = 0.0
            self.lbl_motion.setText("Vehicle: STATIONARY / PARKED (score: 0.0)")
            self.lbl_motion.setStyleSheet("font-size: 12px; color: #f59e0b; font-weight: bold; border: none;")

        if self.active_session is not None:
            evt_type = "simulated_driving_enabled" if self.simulate_driving else "simulated_driving_disabled"
            Event.log(self.active_session.id, evt_type, 0.0, 0.0, f"Simulated Driving: {self.simulate_driving}")

    def update_threshold_labels(self):
        """Update textual threshold readouts."""
        ear_thr = self.config.get("thresholds", {}).get("ear_threshold", 0.21)
        mar_thr = self.config.get("thresholds", {}).get("mar_threshold", 0.60)
        self.lbl_ear_threshold.setText(f"Alert Threshold: {ear_thr:.2f}")
        self.lbl_mar_threshold.setText(f"Yawn Threshold: {mar_thr:.2f}")

    def check_battery_health(self):
        """Periodically check device battery power status."""
        bat = get_battery_status(self.config.get("battery", {}).get("low_battery_threshold", 20))
        if bat["has_battery"]:
            color = "#ef4444" if bat["is_low"] else "#10b981"
            self.lbl_battery.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: bold;")
            self.lbl_battery.setText(bat["status_text"])
            
            if bat["is_low"]:
                self.health_banner.show()
                self.health_banner.setText("WARNING: Laptop battery is low (<20%). Connect charger to avoid session loss.")
                if self.active_session is not None:
                    Event.log(self.active_session.id, "low_battery_warning", 0.0, 0.0, f"{bat['percent']}%")
            else:
                if not self.is_low_light_state:
                    self.health_banner.hide()
        else:
            self.lbl_battery.setText("Power: AC")

    def start_camera(self):
        """Initialize and start capture thread with disconnect callbacks."""
        camera_idx = self.config.get("camera", {}).get("index", 0)
        
        self.capture_thread = CaptureThread(camera_idx)
        self.capture_thread.frame_captured.connect(self.on_frame_captured)
        self.capture_thread.status_changed.connect(self.on_status_changed)
        self.capture_thread.start()

        # Start tracking Session in database
        self.active_session = Session.create(
            user_label="Primary-User",
            baseline_ear=self.config.get("thresholds", {}).get("ear_threshold", 0.21)
        )
        self.session_seconds = 0
        self.session_timer.start(1000)
        self.shared_state.set_monitoring(True)

    def stop_camera(self):
        """Gracefully stop capture thread and persist session end timestamp."""
        self.session_timer.stop()
        self.shared_state.set_monitoring(False)
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
            
        if self.active_session is not None:
            self.active_session.end()
            self.active_session = None

        self.camera_feed_lbl.setText("Camera Offline")
        self.lbl_status.setText("MONITOR INACTIVE")
        self.status_card.setStyleSheet("background-color: #27272a; border-radius: 12px;")
        self.alert_manager.stop_alarm()
        self.dismiss_btn.setEnabled(False)
        self.admin_btn.setEnabled(False)
        self.shared_state.update_metrics(state="MONITOR INACTIVE", fatigue_score=0, ear=0, mar=0)

    def toggle_monitoring(self):
        """Start or stop webcam monitoring."""
        if self.capture_thread is not None:
            self.stop_camera()
            self.camera_toggle_btn.setText("Start Monitoring")
        else:
            self.start_camera()
            self.camera_toggle_btn.setText("Stop Monitoring")

    @pyqtSlot(object)
    def on_frame_captured(self, frame):
        """Process incoming video frames, execute multi-signal analysis, and draw HUD."""
        h, w, _ = frame.shape
        
        # 1. Low-Light Adaptive Preprocessing (CLAHE / Gamma & Glare Reduction)
        processed_frame, is_low_light, luminance = preprocess_frame(frame, self.config)
        self.is_low_light_state = is_low_light
        self.current_luminance = luminance

        # 2. Multi-Face Landmark Detection & Primary Driver Isolation
        roi_cfg = self.config.get("multi_face", {}).get("roi", None)
        primary_landmarks, all_faces = self.detector.detect_multi_faces(processed_frame, roi_cfg)
        has_face = primary_landmarks is not None

        # 3. Feature Extraction (EAR, MAR, Head Pose)
        ear = 0.0
        mar = 0.0
        pitch, yaw, roll = 0.0, 0.0, 0.0
        
        if has_face:
            ear = calculate_ear(primary_landmarks, w, h)
            mar = calculate_mar(primary_landmarks, w, h)
            pitch, yaw, roll = calculate_head_pose(primary_landmarks, w, h)
            
            if self.calibration_active:
                self.calibration_frames.append(ear)

        # 4. FSM Classifier Evaluation (Multi-Signal: EAR + Yawn Frequency + Posture)
        # Note: Only primary driver landmarks are processed for fatigue classification!
        state, score, events = self.classifier.process_frame(has_face, ear, mar, pitch, yaw, roll)
        
        # 4b. Vehicle Motion Detection (Driving vs Parked/Stationary)
        motion_cfg = self.config.get("motion", {})
        motion_enabled = motion_cfg.get("enabled", True)
        require_motion_for_alert = motion_cfg.get("require_motion_for_alert", True)

        if self.simulate_driving:
            self.is_vehicle_moving = True
            self.current_motion_score = 99.0
            is_driving = True
            self.lbl_motion.setText("Vehicle: 🚗 DRIVING (TEST SIMULATION)")
            self.lbl_motion.setStyleSheet("font-size: 12px; color: #10b981; font-weight: bold; border: none;")
        elif motion_enabled:
            is_moving, motion_score = self.motion_detector.update(frame)
            self.is_vehicle_moving = is_moving
            self.current_motion_score = motion_score
            is_driving = is_moving or (not require_motion_for_alert)

            # Update motion label in telemetry panel
            if self.is_vehicle_moving:
                self.lbl_motion.setText(f"Vehicle: MOVING / DRIVING (score: {self.current_motion_score:.1f})")
                self.lbl_motion.setStyleSheet("font-size: 12px; color: #10b981; font-weight: bold; border: none;")
            else:
                self.lbl_motion.setText(f"Vehicle: STATIONARY / PARKED (score: {self.current_motion_score:.1f})")
                self.lbl_motion.setStyleSheet("font-size: 12px; color: #f59e0b; font-weight: bold; border: none;")
        else:
            self.is_vehicle_moving = True
            self.current_motion_score = 0.0
            is_driving = True
            self.lbl_motion.setText("Vehicle: DRIVING (Motion Gating Disabled)")
            self.lbl_motion.setStyleSheet("font-size: 12px; color: #10b981; font-weight: bold; border: none;")

        # 5. UI Updates
        self.update_status_display(state, score, is_driving)
        self.update_metrics_bars(ear, mar, score, pitch, yaw)

        # 6. Alarm Management (Alert ONLY if DRIVER is sleeping AND vehicle is DRIVING)
        if state == State.DROWSY_ALERT:
            if is_driving:
                self.alert_manager.play_alarm()
                self.dismiss_btn.setEnabled(True)
                self.admin_btn.setEnabled(True)
            else:
                # Vehicle is parked / not moving — mute buzzer alert
                self.alert_manager.stop_alarm()
                self.dismiss_btn.setEnabled(False)
                self.admin_btn.setEnabled(False)
        else:
            if self.alert_manager.is_playing and not self.alert_manager.sound_enabled:
                pass
            elif state != State.DROWSY_ALERT:
                self.alert_manager.stop_alarm()
                self.alert_manager.sound_enabled = True
                self.dismiss_btn.setEnabled(False)
                self.admin_btn.setEnabled(False)

        # 6b. Remote Admin Buzzer Command Processing
        cmd, remote_ip = self.shared_state.consume_buzzer_command()
        if cmd == BuzzerCommand.MUTE:
            self.alert_manager.stop_alarm()
            self.alert_manager.sound_enabled = False
            self.classifier.reset()
            self.update_status_display(State.AWAKE, 0.0, is_driving)
            self.dismiss_btn.setEnabled(False)
            self.admin_btn.setEnabled(False)
            if self.active_session is not None:
                Event.log(self.active_session.id, "admin_remote_override", ear, mar, f"Remote Mute from {remote_ip}")
        elif cmd == BuzzerCommand.UNMUTE:
            self.alert_manager.sound_enabled = True
            if self.active_session is not None:
                Event.log(self.active_session.id, "admin_remote_unmute", ear, mar, f"Remote Unmute from {remote_ip}")

        # 7. Push metrics to SharedState for remote admin dashboard
        self.shared_state.update_metrics(
            state=state.value,
            fatigue_score=score,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_buzzer_playing=self.alert_manager.is_playing,
            is_low_light=is_low_light,
            luminance=luminance,
            active_yawn_count=self.classifier.active_yawn_count,
            session_seconds=self.session_seconds,
            blink_rate=len(self.classifier.blink_timestamps),
            faces_detected=len(all_faces),
            is_vehicle_moving=self.is_vehicle_moving,
            motion_score=self.current_motion_score,
            is_simulated_driving=self.simulate_driving
        )

        # 8. Database Telemetry Logging
        if self.active_session is not None:
            self.log_throttle_counter += 1
            for evt in events:
                Event.log(self.active_session.id, evt, ear, mar)
                self.shared_state.push_event(evt, time.strftime('%Y-%m-%dT%H:%M:%S'))
                
            if self.log_throttle_counter >= 30:
                self.log_throttle_counter = 0
                Event.log(self.active_session.id, state.value.lower(), ear, mar)

        # 9. Render HUD Annotations onto Frame
        # A. Low-light badge
        if is_low_light:
            cv2.putText(
                frame, "LOW LIGHT: CLAHE ACTIVE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA
            )

        # B. Vehicle Motion Status Badge
        v_status_text = "VEHICLE: DRIVING" if self.is_vehicle_moving else "VEHICLE: PARKED / STOPPED"
        v_status_color = (0, 255, 0) if self.is_vehicle_moving else (0, 165, 255)
        cv2.putText(
            frame, v_status_text, (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, v_status_color, 2, cv2.LINE_AA
        )

        # C. Multi-Face Passenger Bounding Boxes (Driver monitored, Passengers ignored for alerts)
        if self.config.get("ui", {}).get("show_passengers", True):
            for face in all_faces:
                bx, by, bw, bh = face["bbox"]
                if face["is_primary"]:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(frame, "DRIVER (MONITORED)", (bx, max(20, by - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (128, 128, 128), 1)
                    cv2.putText(frame, "PASSENGER (NO ALERTS)", (bx, max(20, by - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1, cv2.LINE_AA)

        # D. Parked Drowsiness Notice
        if state == State.DROWSY_ALERT and not is_driving:
            cv2.putText(
                frame, "DRIVER DROWSY (VEHICLE PARKED - ALARM MUTED)", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA
            )

        # E. Primary Landmark Mesh Points
        if has_face and self.config.get("ui", {}).get("show_landmarks", True):
            self.draw_landmarks(frame, primary_landmarks, w, h)

        self.display_image(frame)

    def draw_landmarks(self, frame, landmarks, w, h):
        """Draw facial landmark keypoints on frame."""
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        inner_lips = [78, 81, 13, 311, 308, 317, 14, 87]
        
        for idx in left_eye + right_eye:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            
        for idx in inner_lips:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

    def display_image(self, frame):
        """Convert BGR cv2 frame to QImage and draw to QLabel safely with zero lag."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        lbl_w = self.camera_feed_lbl.width()
        lbl_h = self.camera_feed_lbl.height()
        if lbl_w > 10 and lbl_h > 10:
            pixmap = QPixmap.fromImage(qt_img).scaled(
                lbl_w, 
                lbl_h, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            self.camera_feed_lbl.setPixmap(pixmap)
        else:
            self.camera_feed_lbl.setPixmap(QPixmap.fromImage(qt_img))

    def update_status_display(self, state: State, score: float, is_driving: bool = True):
        """Update status label and header styling."""
        if state == State.AWAKE:
            self.lbl_status.setText("AWAKE")
            self.status_card.setStyleSheet("background-color: #059669; border-radius: 12px;")
        elif state == State.DROWSY_WARNING:
            self.lbl_status.setText("DROWSY WARNING")
            self.status_card.setStyleSheet("background-color: #d97706; border-radius: 12px;")
        elif state == State.DROWSY_ALERT:
            if is_driving:
                self.lbl_status.setText("DROWSY ALERT!")
                self.status_card.setStyleSheet("background-color: #dc2626; border-radius: 12px;")
            else:
                self.lbl_status.setText("DROWSY (PARKED - MUTED)")
                self.status_card.setStyleSheet("background-color: #d97706; border-radius: 12px;")
        elif state == State.NO_FACE:
            self.lbl_status.setText("NO DRIVER DETECTED")
            self.status_card.setStyleSheet("background-color: #4b5563; border-radius: 12px;")

    def update_metrics_bars(self, ear: float, mar: float, score: float, pitch: float, yaw: float):
        """Update numerical progress gauges and posture labels."""
        ear_percent = int(min(100, max(0, (ear / 0.40) * 100)))
        self.ear_bar.setValue(ear_percent)
        self.ear_bar.setFormat(f"{ear:.3f}")

        mar_percent = int(min(100, max(0, (mar / 0.80) * 100)))
        self.mar_bar.setValue(mar_percent)
        self.mar_bar.setFormat(f"{mar:.3f}")

        self.fatigue_bar.setValue(int(score))
        
        # Yawn frequency readout
        limit = self.config.get("thresholds", {}).get("yawn_frequency_alert_threshold", 3)
        self.lbl_yawn_freq.setText(f"Rolling 5m Yawns: {self.classifier.active_yawn_count} / {limit}")
        
        # Head pose angles
        self.lbl_head_pose.setText(f"Head Pose: Pitch {pitch:.1f}° | Yaw {yaw:.1f}°")

    def dismiss_alert(self):
        """Silence alarm via interactive cognitive math puzzle."""
        dialog = MathPuzzleDialog(self)
        if dialog.exec() == MathPuzzleDialog.DialogCode.Accepted:
            self.alert_manager.stop_alarm()
            self.alert_manager.sound_enabled = True
            self.dismiss_btn.setEnabled(False)
            self.admin_btn.setEnabled(False)
            self.classifier.reset()
            self.update_status_display(State.AWAKE, 0.0)
            if self.active_session is not None:
                Event.log(self.active_session.id, "user_dismiss_puzzle", 0.0, 0.0, "Math Puzzle Solved")

    def admin_override(self):
        """Silence active alarm via Admin PIN authentication."""
        pin = str(self.config.get("admin", {}).get("pin", "1234"))
        dialog = AdminOverrideDialog(config_pin=pin, parent=self)
        if dialog.exec() == AdminOverrideDialog.DialogCode.Accepted:
            self.alert_manager.stop_alarm()
            self.alert_manager.sound_enabled = True
            self.dismiss_btn.setEnabled(False)
            self.admin_btn.setEnabled(False)
            self.classifier.reset()
            self.update_status_display(State.AWAKE, 0.0)
            
            # Log admin override event to database for auditability
            if self.active_session is not None:
                Event.log(self.active_session.id, "admin_override", 0.0, 0.0, "PIN Authenticated Override")
            QMessageBox.information(self, "Override Success", "Alarm silenced and logged by Administrator.")

    @pyqtSlot(str)
    def on_status_changed(self, status_msg):
        """Handle webcam disconnect announcements and trigger graceful recovery."""
        if "Disconnected" in status_msg:
            self.camera_feed_lbl.setText("Camera Disconnected")
            self.health_banner.show()
            self.health_banner.setText("CRITICAL: Camera connection lost. Auto-saving session data...")
            if self.active_session is not None:
                Event.log(self.active_session.id, "camera_disconnect", 0.0, 0.0, "Hardware Disconnect")
                self.active_session.end()
        elif "Connecting" in status_msg:
            self.camera_feed_lbl.setText(status_msg)

    def tick_session_timer(self):
        """Increment session timer display."""
        self.session_seconds += 1
        m, s = divmod(self.session_seconds, 60)
        self.lbl_timer.setText(f"Duration: {m:02d}:{s:02d}")

    # --- Calibration & User Profiles ---
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
        self.calibration_banner.setText(f"Calibrating baseline... Look straight at camera.\nCountdown: {self.calibration_countdown}s")
        self.calibration_timer.start(1000)

    def tick_calibration(self):
        """Tick down calibration timer."""
        self.calibration_countdown -= 1
        if self.calibration_countdown > 0:
            self.calibration_banner.setText(f"Calibrating baseline... Look straight at camera.\nCountdown: {self.calibration_countdown}s")
        else:
            self.calibration_timer.stop()
            self.calibration_active = False
            self.calibrate_btn.setEnabled(True)
            self.calibration_banner.hide()
            
            if len(self.calibration_frames) > 50:
                mean_ear = float(np.mean(self.calibration_frames))
                new_threshold = round(mean_ear * 0.70, 3)
                
                self.config["thresholds"]["ear_threshold"] = new_threshold
                self.classifier.update_config(self.config)
                self.update_threshold_labels()
                
                if self.active_session is not None:
                    self.active_session.baseline_ear = mean_ear
                
                QMessageBox.information(
                    self, "Calibration Completed",
                    f"Baseline EAR: {mean_ear:.3f}\nPersonal Alert Threshold: {new_threshold:.3f}\n"
                    "You can save this profile in the Profiles menu."
                )
            else:
                QMessageBox.warning(self, "Calibration Incomplete", "Not enough face samples captured. Please retry.")

    def open_profiles(self):
        """Open persistent user profile manager dialog."""
        dlg = ProfileManagerDialog(self.config, self)
        def on_profile_loaded(profile: UserProfile):
            self.config["thresholds"]["ear_threshold"] = profile.ear_threshold
            self.config["thresholds"]["mar_threshold"] = profile.mar_threshold
            self.classifier.update_config(self.config)
            self.update_threshold_labels()
            QMessageBox.information(self, "Profile Applied", f"Loaded '{profile.name}' calibration profile!")
            
        dlg.profile_loaded.connect(on_profile_loaded)
        dlg.exec()

    def open_settings(self):
        """Open settings dialog."""
        dlg = SettingsWindow(self.config, self)
        def on_saved(new_config):
            self.config = new_config
            self.classifier.update_config(self.config)
            self.alert_manager.update_config(self.config)
            self.update_threshold_labels()
            self.simulate_driving = self.config.get("motion", {}).get("simulate_driving", False)
            self.update_simulated_driving_button()
            camera_idx = self.config.get("camera", {}).get("index", 0)
            if self.capture_thread is not None and self.capture_thread.camera_index != camera_idx:
                self.stop_camera()
                self.start_camera()

        dlg.config_saved.connect(on_saved)
        dlg.exec()

    def open_summary(self):
        """Open historical session analytics."""
        dlg = SummaryWindow(self)
        dlg.exec()

    def open_wifi_access(self):
        """Open Wi-Fi & Mobile QR Code Access Dialog."""
        admin_port = self.config.get("admin", {}).get("remote_port", 8080)
        dlg = WiFiAccessDialog(port=admin_port, parent=self)
        dlg.exec()

    def closeEvent(self, event):
        """Clean up threads, timers, and alerts when window is closed."""
        try:
            self.battery_timer.stop()
            self.calibration_timer.stop()
            self.stop_camera()
            self.alert_manager.stop_alarm()
        except Exception:
            pass
        event.accept()
