"""Settings Window.

Allows configuration of fatigue detection thresholds, camera selection,
low-light preprocessing, posture analysis, and admin override PIN.
"""

import yaml
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox, QLineEdit, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any

class SettingsWindow(QDialog):
    """Dialog settings panel to edit thresholds and feature parameters."""
    config_saved = pyqtSignal(dict)
    calibration_requested = pyqtSignal()

    def __init__(self, config: Dict[str, Any], parent=None):
        """Initialize settings dialog.
        
        Args:
            config: Active configuration dictionary.
            parent: Parent QWidget.
        """
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("System Configuration & Detection Settings")
        self.resize(520, 620)
        self.setObjectName("settings_dialog")
        self.init_ui()

    def init_ui(self):
        """Create dialog widgets with scrollable form layouts."""
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(10, 10, 10, 10)

        # Scroll container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(12)

        title_label = QLabel("System Configuration")
        title_label.setObjectName("header")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # --- 1. Camera & Hardware ---
        camera_group = QGroupBox("Camera & Hardware")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_combo = QComboBox()
        for i in range(4):
            self.camera_combo.addItem(f"Webcam {i}", i)
        camera_index = self.config.get("camera", {}).get("index", 0)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData(camera_index))
        camera_layout.addRow("Active Camera Index:", self.camera_combo)
        main_layout.addWidget(camera_group)

        # --- 2. Fatigue Detection Thresholds ---
        threshold_group = QGroupBox("Eye & Yawn Thresholds")
        threshold_layout = QFormLayout(threshold_group)

        self.ear_threshold_spin = QDoubleSpinBox()
        self.ear_threshold_spin.setRange(0.10, 0.40)
        self.ear_threshold_spin.setSingleStep(0.01)
        self.ear_threshold_spin.setDecimals(2)
        self.ear_threshold_spin.setValue(self.config.get("thresholds", {}).get("ear_threshold", 0.21))
        
        self.ear_frames_spin = QSpinBox()
        self.ear_frames_spin.setRange(5, 100)
        self.ear_frames_spin.setValue(self.config.get("thresholds", {}).get("ear_consec_frames", 20))

        self.mar_threshold_spin = QDoubleSpinBox()
        self.mar_threshold_spin.setRange(0.30, 0.90)
        self.mar_threshold_spin.setSingleStep(0.02)
        self.mar_threshold_spin.setDecimals(2)
        self.mar_threshold_spin.setValue(self.config.get("thresholds", {}).get("mar_threshold", 0.60))

        self.mar_frames_spin = QSpinBox()
        self.mar_frames_spin.setRange(5, 100)
        self.mar_frames_spin.setValue(self.config.get("thresholds", {}).get("mar_consec_frames", 15))

        threshold_layout.addRow("Eye Closure Threshold (EAR):", self.ear_threshold_spin)
        threshold_layout.addRow("Eye Closed Sustained Frames:", self.ear_frames_spin)
        threshold_layout.addRow("Yawn Opening Threshold (MAR):", self.mar_threshold_spin)
        threshold_layout.addRow("Yawn Sustained Frames:", self.mar_frames_spin)
        main_layout.addWidget(threshold_group)

        # --- 3. Yawn Frequency Tracking ---
        yawn_freq_group = QGroupBox("Rolling Yawn Frequency")
        yawn_layout = QFormLayout(yawn_freq_group)

        self.yawn_window_spin = QSpinBox()
        self.yawn_window_spin.setRange(60, 1800)
        self.yawn_window_spin.setSuffix(" sec")
        self.yawn_window_spin.setValue(self.config.get("thresholds", {}).get("yawn_window_seconds", 300))

        self.yawn_limit_spin = QSpinBox()
        self.yawn_limit_spin.setRange(1, 20)
        self.yawn_limit_spin.setValue(self.config.get("thresholds", {}).get("yawn_frequency_alert_threshold", 3))

        yawn_layout.addRow("Rolling Window Duration:", self.yawn_window_spin)
        yawn_layout.addRow("Yawn Count Escalation Limit:", self.yawn_limit_spin)
        main_layout.addWidget(yawn_freq_group)

        # --- 4. Low-Light & CLAHE Enhancement ---
        low_light_group = QGroupBox("Low-Light Robustness (CLAHE / Gamma)")
        ll_layout = QFormLayout(low_light_group)

        self.ll_enable_check = QCheckBox("Enable Adaptive Low-Light Enhancement")
        self.ll_enable_check.setChecked(self.config.get("low_light", {}).get("enabled", True))
        ll_layout.addRow(self.ll_enable_check)

        self.ll_thresh_spin = QDoubleSpinBox()
        self.ll_thresh_spin.setRange(10.0, 150.0)
        self.ll_thresh_spin.setValue(self.config.get("low_light", {}).get("brightness_threshold", 65.0))
        ll_layout.addRow("Luminance Trigger Threshold:", self.ll_thresh_spin)

        self.ll_gamma_spin = QDoubleSpinBox()
        self.ll_gamma_spin.setRange(0.2, 1.0)
        self.ll_gamma_spin.setSingleStep(0.05)
        self.ll_gamma_spin.setValue(self.config.get("low_light", {}).get("gamma", 0.6))
        ll_layout.addRow("Gamma Brightening Factor:", self.ll_gamma_spin)

        main_layout.addWidget(low_light_group)

        # --- 5. Posture & Head Nodding ---
        posture_group = QGroupBox("Posture & Head Pose Estimation")
        posture_layout = QFormLayout(posture_group)

        self.posture_enable_check = QCheckBox("Enable Head Pose / Nodding Analysis")
        self.posture_enable_check.setChecked(self.config.get("posture", {}).get("enabled", True))
        posture_layout.addRow(self.posture_enable_check)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-45.0, 0.0)
        self.pitch_spin.setSuffix(" deg")
        self.pitch_spin.setValue(self.config.get("posture", {}).get("pitch_nod_threshold", -18.0))
        posture_layout.addRow("Head Droop Nod Threshold (Pitch):", self.pitch_spin)

        main_layout.addWidget(posture_group)

        # --- 6. Vehicle Motion & Driving Gating ---
        motion_group = QGroupBox("Vehicle Motion & Driving Gating")
        motion_layout = QFormLayout(motion_group)

        self.motion_enable_check = QCheckBox("Enable Vehicle Motion Detection")
        self.motion_enable_check.setChecked(self.config.get("motion", {}).get("enabled", True))
        motion_layout.addRow(self.motion_enable_check)

        self.motion_gate_check = QCheckBox("Only Alert When Vehicle is Moving (Mute when Parked)")
        self.motion_gate_check.setChecked(self.config.get("motion", {}).get("require_motion_for_alert", True))
        motion_layout.addRow(self.motion_gate_check)

        self.motion_simulate_check = QCheckBox("🚗 Force / Simulate Driving Mode (Desk Testing without Motion)")
        self.motion_simulate_check.setChecked(self.config.get("motion", {}).get("simulate_driving", False))
        self.motion_simulate_check.setToolTip("Overrides motion gating to simulate active driving while stationary at a desk")
        motion_layout.addRow(self.motion_simulate_check)

        self.motion_thresh_spin = QDoubleSpinBox()
        self.motion_thresh_spin.setRange(1.0, 30.0)
        self.motion_thresh_spin.setSingleStep(0.5)
        self.motion_thresh_spin.setValue(self.config.get("motion", {}).get("motion_threshold", 4.0))
        motion_layout.addRow("Motion Sensitivity Threshold:", self.motion_thresh_spin)

        main_layout.addWidget(motion_group)

        # --- 7. Admin & Security Controls ---
        admin_group = QGroupBox("Admin Supervisor Override")
        admin_layout = QFormLayout(admin_group)

        self.admin_pin_input = QLineEdit()
        self.admin_pin_input.setPlaceholderText("Admin PIN")
        self.admin_pin_input.setText(str(self.config.get("admin", {}).get("pin", "1234")))
        admin_layout.addRow("Admin Override PIN:", self.admin_pin_input)

        self.remote_enable_check = QCheckBox("Enable Remote Admin Web Panel (LAN)")
        self.remote_enable_check.setChecked(self.config.get("admin", {}).get("remote_enabled", True))
        admin_layout.addRow(self.remote_enable_check)

        self.remote_port_spin = QSpinBox()
        self.remote_port_spin.setRange(1024, 65535)
        self.remote_port_spin.setValue(self.config.get("admin", {}).get("remote_port", 8080))
        admin_layout.addRow("Remote Panel Port:", self.remote_port_spin)

        main_layout.addWidget(admin_group)

        # --- 7. Preferences ---
        ui_group = QGroupBox("Display & Audio Preferences")
        ui_layout = QVBoxLayout(ui_group)
        
        self.overlay_check = QCheckBox("Render Facial Landmark Dot Overlay")
        self.overlay_check.setChecked(self.config.get("ui", {}).get("show_landmarks", True))
        
        self.passengers_check = QCheckBox("Render Passenger / Bystander Bounding Boxes")
        self.passengers_check.setChecked(self.config.get("ui", {}).get("show_passengers", True))

        self.volume_check = QCheckBox("Enable Alarm Sound Alerts")
        vol = self.config.get("alerts", {}).get("volume", 0.8)
        self.volume_check.setChecked(vol > 0)
        
        ui_layout.addWidget(self.overlay_check)
        ui_layout.addWidget(self.passengers_check)
        ui_layout.addWidget(self.volume_check)
        main_layout.addWidget(ui_group)

        scroll.setWidget(container)
        dialog_layout.addWidget(scroll)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self.save_config)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        dialog_layout.addLayout(btn_layout)

    def save_config(self):
        """Parse widgets and save to config.yaml."""
        # Camera
        self.config.setdefault("camera", {})["index"] = self.camera_combo.currentData()
        
        # Thresholds
        thr = self.config.setdefault("thresholds", {})
        thr["ear_threshold"] = round(self.ear_threshold_spin.value(), 3)
        thr["ear_consec_frames"] = self.ear_frames_spin.value()
        thr["mar_threshold"] = round(self.mar_threshold_spin.value(), 3)
        thr["mar_consec_frames"] = self.mar_frames_spin.value()
        thr["yawn_window_seconds"] = self.yawn_window_spin.value()
        thr["yawn_frequency_alert_threshold"] = self.yawn_limit_spin.value()

        # Low light
        ll = self.config.setdefault("low_light", {})
        ll["enabled"] = self.ll_enable_check.isChecked()
        ll["brightness_threshold"] = self.ll_thresh_spin.value()
        ll["gamma"] = self.ll_gamma_spin.value()

        # Posture
        posture = self.config.setdefault("posture", {})
        posture["enabled"] = self.posture_enable_check.isChecked()
        posture["pitch_nod_threshold"] = self.pitch_spin.value()

        # Motion
        motion = self.config.setdefault("motion", {})
        motion["enabled"] = self.motion_enable_check.isChecked()
        motion["require_motion_for_alert"] = self.motion_gate_check.isChecked()
        motion["simulate_driving"] = self.motion_simulate_check.isChecked()
        motion["motion_threshold"] = self.motion_thresh_spin.value()

        # Admin
        admin = self.config.setdefault("admin", {})
        admin["pin"] = self.admin_pin_input.text().strip()
        admin["remote_enabled"] = self.remote_enable_check.isChecked()
        admin["remote_port"] = self.remote_port_spin.value()

        # UI & Alerts
        ui = self.config.setdefault("ui", {})
        ui["show_landmarks"] = self.overlay_check.isChecked()
        ui["show_passengers"] = self.passengers_check.isChecked()
        self.config.setdefault("alerts", {})["volume"] = 0.8 if self.volume_check.isChecked() else 0.0

        # Save to file
        try:
            with open("config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.config_saved.emit(self.config)
            self.accept()
        except Exception as e:
            print(f"Error saving config.yaml: {e}")
            self.reject()
