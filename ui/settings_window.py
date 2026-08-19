"""Settings Window.

Allows configuration of detection thresholds, camera selection, and baseline calibration.
"""

import yaml
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QCheckBox, QPushButton, QComboBox, QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any

class SettingsWindow(QDialog):
    """Dialog settings panel to edit thresholds and run calibration."""
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
        self.setWindowTitle("Detection Settings")
        self.setMinimumWidth(450)
        self.setObjectName("settings_dialog")
        self.init_ui()

    def init_ui(self):
        """Create dialog widgets and form layouts."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        title_label = QLabel("Configuration Panel")
        title_label.setObjectName("header")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Group: Camera Settings ---
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_combo = QComboBox()
        # Scan basic camera indices (0 to 3)
        for i in range(4):
            self.camera_combo.addItem(f"Webcam {i}", i)
        
        camera_index = self.config.get("camera", {}).get("index", 0)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData(camera_index))
        camera_layout.addRow("Active Camera:", self.camera_combo)
        main_layout.addWidget(camera_group)

        # --- Group: Detection Thresholds ---
        threshold_group = QGroupBox("Fatigue Detection Tunables")
        threshold_layout = QFormLayout(threshold_group)

        # EAR threshold double spinbox
        self.ear_threshold_spin = QDoubleSpinBox()
        self.ear_threshold_spin.setRange(0.10, 0.40)
        self.ear_threshold_spin.setSingleStep(0.01)
        self.ear_threshold_spin.setDecimals(2)
        ear_val = self.config.get("thresholds", {}).get("ear_threshold", 0.21)
        self.ear_threshold_spin.setValue(ear_val)
        
        # EAR frames spinbox
        self.ear_frames_spin = QSpinBox()
        self.ear_frames_spin.setRange(5, 100)
        ear_frames = self.config.get("thresholds", {}).get("ear_consec_frames", 20)
        self.ear_frames_spin.setValue(ear_frames)

        # MAR threshold double spinbox
        self.mar_threshold_spin = QDoubleSpinBox()
        self.mar_threshold_spin.setRange(0.30, 0.90)
        self.mar_threshold_spin.setSingleStep(0.02)
        self.mar_threshold_spin.setDecimals(2)
        mar_val = self.config.get("thresholds", {}).get("mar_threshold", 0.6)
        self.mar_threshold_spin.setValue(mar_val)

        # MAR frames spinbox
        self.mar_frames_spin = QSpinBox()
        self.mar_frames_spin.setRange(5, 100)
        mar_frames = self.config.get("thresholds", {}).get("mar_consec_frames", 15)
        self.mar_frames_spin.setValue(mar_frames)

        threshold_layout.addRow("Eye Closure Threshold (EAR):", self.ear_threshold_spin)
        threshold_layout.addRow("Eye Closed Sustained Frames:", self.ear_frames_spin)
        threshold_layout.addRow("Yawn Opening Threshold (MAR):", self.mar_threshold_spin)
        threshold_layout.addRow("Yawn Sustained Frames:", self.mar_frames_spin)
        main_layout.addWidget(threshold_group)

        # --- Calibration Block ---
        calibration_group = QGroupBox("Baseline Calibration")
        calib_layout = QVBoxLayout(calibration_group)
        
        calib_desc = QLabel(
            "Press Calibrate and look straight at the screen. The app will record "
            "your eye profile for 10 seconds to tune your personal EAR thresholds."
        )
        calib_desc.setWordWrap(True)
        calib_desc.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        
        self.calibrate_btn = QPushButton("Start Calibration Scan")
        self.calibrate_btn.setObjectName("secondary_btn")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        
        calib_layout.addWidget(calib_desc)
        calib_layout.addWidget(self.calibrate_btn)
        main_layout.addWidget(calibration_group)

        # --- Group: Visual Settings ---
        ui_group = QGroupBox("Preferences")
        ui_layout = QVBoxLayout(ui_group)
        
        self.overlay_check = QCheckBox("Render Landmark Dot Overlay on Camera Feed")
        self.overlay_check.setChecked(self.config.get("ui", {}).get("show_landmarks", True))
        
        self.volume_check = QCheckBox("Enable Alarm Sound Alerts")
        # Read volume config or simple sound status
        vol = self.config.get("alerts", {}).get("volume", 0.8)
        self.volume_check.setChecked(vol > 0)
        
        ui_layout.addWidget(self.overlay_check)
        ui_layout.addWidget(self.volume_check)
        main_layout.addWidget(ui_group)

        # --- Save/Reset Actions ---
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_config)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

    def start_calibration(self):
        """Emit calibration signal and temporarily disable window."""
        self.calibration_requested.emit()
        self.calibrate_btn.setText("Calibrating in monitor window...")
        self.calibrate_btn.setEnabled(False)

    def update_calibrated_ear(self, new_ear_threshold: float):
        """Callback to receive calibration results from main thread."""
        self.ear_threshold_spin.setValue(new_ear_threshold)
        self.calibrate_btn.setText("Calibration Done! Recalibrate")
        self.calibrate_btn.setEnabled(True)

    def save_config(self):
        """Parse widgets and save to config.yaml."""
        # Update config dict
        self.config["camera"]["index"] = self.camera_combo.currentData()
        
        self.config["thresholds"]["ear_threshold"] = round(self.ear_threshold_spin.value(), 3)
        self.config["thresholds"]["ear_consec_frames"] = self.ear_frames_spin.value()
        
        self.config["thresholds"]["mar_threshold"] = round(self.mar_threshold_spin.value(), 3)
        self.config["thresholds"]["mar_consec_frames"] = self.mar_frames_spin.value()
        
        self.config["ui"]["show_landmarks"] = self.overlay_check.isChecked()
        self.config["alerts"]["volume"] = 0.8 if self.volume_check.isChecked() else 0.0

        # Save to file
        try:
            with open("config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.config_saved.emit(self.config)
            self.accept()
        except Exception as e:
            print(f"Error saving config.yaml: {e}")
            self.reject()
