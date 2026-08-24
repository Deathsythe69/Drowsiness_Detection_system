"""Admin Override Dialog Module.

Provides PIN-authenticated supervisor/admin override to silence
the active buzzer alarm and log the override to the database.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt

class AdminOverrideDialog(QDialog):
    """PIN-gated dialog for supervisor alarm silencing."""

    def __init__(self, config_pin: str = "1234", parent=None):
        """Initialize Admin Override Dialog.
        
        Args:
            config_pin: Required admin PIN.
            parent: Parent Qt widget.
        """
        super().__init__(parent)
        self.config_pin = str(config_pin)
        self.setWindowTitle("Admin Alarm Override")
        self.setFixedSize(360, 220)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.init_ui()

    def init_ui(self):
        """Build the admin PIN verification interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("ADMIN ALARM OVERRIDE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f59e0b;")
        layout.addWidget(title)

        desc = QLabel("Enter the Administrator PIN to silence the active alert. This override will be recorded in the audit log.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #a1a1aa;")
        layout.addWidget(desc)

        self.pin_input = QLineEdit()
        self.pin_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pin_input.setPlaceholderText("Enter Admin PIN")
        self.pin_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pin_input.setStyleSheet("""
            QLineEdit {
                background-color: #18181b;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 8px;
                font-size: 16px;
                color: #ffffff;
                letter-spacing: 4px;
            }
            QLineEdit:focus {
                border-color: #f59e0b;
            }
        """)
        self.pin_input.returnPressed.connect(self.verify_pin)
        layout.addWidget(self.pin_input)

        btn_row = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.submit_btn = QPushButton("Authenticate & Silence")
        self.submit_btn.setStyleSheet("background-color: #d97706; color: white; font-weight: bold;")
        self.submit_btn.clicked.connect(self.verify_pin)
        btn_row.addWidget(self.submit_btn)

        layout.addLayout(btn_row)

    def verify_pin(self):
        """Verify entered PIN against configuration."""
        entered = self.pin_input.text().strip()
        if entered == self.config_pin:
            self.accept()
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Admin PIN. Please try again.")
            self.pin_input.clear()
            self.pin_input.setFocus()
