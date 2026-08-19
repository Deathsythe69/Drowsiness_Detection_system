"""Main Entry Point.

Initializes configuration, local files, applies visual themes,
and launches the PyQt6 GUI application loop.
"""

import sys
import os
import yaml
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.alerts import generate_default_alarm

DEFAULT_CONFIG = {
    "camera": {
        "index": 0
    },
    "thresholds": {
        "ear_threshold": 0.21,
        "ear_consec_frames": 20,
        "mar_threshold": 0.60,
        "mar_consec_frames": 15,
        "fatigue_warning_score": 40.0,
        "fatigue_alert_score": 75.0
    },
    "alerts": {
        "sound_file": "assets/alarm.wav",
        "volume": 0.8
    },
    "ui": {
        "show_landmarks": True,
        "theme": "dark"
    }
}

QSS_STYLESHEET = """
/* Premium Dark theme stylesheet */

QWidget {
    background-color: #121214;
    color: #e4e4e7;
    font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
    font-size: 13px;
}

QDialog {
    background-color: #121214;
    border: 1px solid #2e2e33;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #27272a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 15px;
    color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}

QLabel#header {
    font-size: 22px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 5px;
}

QLabel#subheader {
    font-size: 13px;
    color: #a1a1aa;
}

QFrame#card {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 10px;
}

QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #3b82f6;
}

QPushButton:pressed {
    background-color: #1d4ed8;
}

QPushButton#secondary_btn {
    background-color: #27272a;
    color: #e4e4e7;
    border: 1px solid #3f3f46;
}

QPushButton#secondary_btn:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}

QPushButton#secondary_btn:pressed {
    background-color: #18181b;
}

QPushButton#danger_btn {
    background-color: #dc2626;
    color: #ffffff;
}

QPushButton#danger_btn:hover {
    background-color: #ef4444;
}

QPushButton#danger_btn:pressed {
    background-color: #b91c1c;
}

QPushButton:disabled {
    background-color: #27272a;
    color: #71717a;
    border: 1px solid #18181b;
}

QProgressBar {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    text-align: center;
    font-weight: bold;
    color: #ffffff;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}

QComboBox {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 6px 12px;
    color: #ffffff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 0px;
}

QDoubleSpinBox, QSpinBox {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 5px;
    color: #ffffff;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #27272a;
    border-radius: 4px;
    background-color: #18181b;
}

QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #3b82f6;
}

QScrollBar:vertical {
    border: none;
    background: #121214;
    width: 10px;
    margin: 0px 0 0px 0;
}

QScrollBar::handle:vertical {
    background: #27272a;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3f3f46;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

def load_or_create_config() -> dict:
    """Load config.yaml if present, otherwise create with default values.
    
    Returns:
        dict: Configuration values.
    """
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        try:
            with open(config_file, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            return DEFAULT_CONFIG
        except Exception as e:
            print(f"Error creating default config.yaml: {e}")
            return DEFAULT_CONFIG
            
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # Ensure basic structure is merged in case user deleted keys
            for key, val in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = val
            return config
    except Exception as e:
        print(f"Error reading config.yaml, using defaults: {e}")
        return DEFAULT_CONFIG

def main():
    """Initializes assets, database, styling, and starts the PyQt window."""
    # 1. Load config
    config = load_or_create_config()

    # 2. Check and generate alarm sound asset
    sound_path = config.get("alerts", {}).get("sound_file", "assets/alarm.wav")
    try:
        generate_default_alarm(sound_path)
    except Exception as e:
        print(f"Error synthesizing sound: {e}")

    # 3. Start PyQt Application
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS_STYLESHEET)
    
    window = MainWindow(config)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
