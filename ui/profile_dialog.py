"""User Profile & Calibration Manager Dialog.

Allows users to manage, save, and load personalized eye/mouth calibration
profiles from SQLite to avoid repetitive calibration across sessions.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QMessageBox, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from storage.models import UserProfile

class ProfileManagerDialog(QDialog):
    """Dialog for managing persistent calibration profiles."""

    profile_loaded = pyqtSignal(object)  # Emits selected UserProfile

    def __init__(self, current_config: dict, parent=None):
        """Initialize Profile Manager.
        
        Args:
            current_config: System configuration dictionary.
            parent: Parent Qt widget.
        """
        super().__init__(parent)
        self.config = current_config
        self.setWindowTitle("User Calibration Profiles")
        self.setFixedSize(520, 440)
        self.init_ui()
        self.load_profiles_list()

    def init_ui(self):
        """Build profile management interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        header = QLabel("Saved Calibration Profiles")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)

        desc = QLabel("Select a stored profile to apply your personalized eye/mouth thresholds immediately.")
        desc.setStyleSheet("font-size: 12px; color: #a1a1aa;")
        layout.addWidget(desc)

        # Profile list widget
        self.profile_list = QListWidget()
        self.profile_list.setStyleSheet("""
            QListWidget {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 5px;
                color: #e4e4e7;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #27272a;
            }
            QListWidget::item:selected {
                background-color: #2563eb;
                color: #ffffff;
                border-radius: 4px;
            }
        """)
        self.profile_list.currentItemChanged.connect(self.on_profile_selected)
        layout.addWidget(self.profile_list)

        # Details groupbox
        details_box = QGroupBox("Selected Profile Details")
        grid = QGridLayout(details_box)
        grid.setSpacing(6)

        grid.addWidget(QLabel("Profile Name:"), 0, 0)
        self.lbl_name = QLabel("-")
        self.lbl_name.setStyleSheet("font-weight: bold; color: #60a5fa;")
        grid.addWidget(self.lbl_name, 0, 1)

        grid.addWidget(QLabel("Baseline EAR / Alert Threshold:"), 1, 0)
        self.lbl_ear = QLabel("-")
        grid.addWidget(self.lbl_ear, 1, 1)

        grid.addWidget(QLabel("Baseline MAR / Yawn Threshold:"), 2, 0)
        self.lbl_mar = QLabel("-")
        grid.addWidget(self.lbl_mar, 2, 1)

        grid.addWidget(QLabel("Recorded Blink Rate:"), 3, 0)
        self.lbl_blink = QLabel("-")
        grid.addWidget(self.lbl_blink, 3, 1)

        layout.addWidget(details_box)

        # Save current settings as profile section
        save_row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("New profile name (e.g. Debasis_Glasses)")
        self.name_input.setStyleSheet("""
            background-color: #18181b;
            border: 1px solid #3f3f46;
            border-radius: 6px;
            padding: 6px;
            color: #ffffff;
        """)
        save_row.addWidget(self.name_input)

        self.save_btn = QPushButton("Save Current as Profile")
        self.save_btn.setObjectName("secondary_btn")
        self.save_btn.clicked.connect(self.save_current_profile)
        save_row.addWidget(self.save_btn)
        layout.addLayout(save_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setObjectName("danger_btn")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_selected_profile)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch()

        self.load_btn = QPushButton("Apply Selected Profile")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold;")
        self.load_btn.clicked.connect(self.apply_profile)
        btn_row.addWidget(self.load_btn)

        layout.addLayout(btn_row)

    def load_profiles_list(self):
        """Fetch profiles from database and populate list."""
        self.profile_list.clear()
        self.profiles = UserProfile.get_all()
        
        for p in self.profiles:
            item = QListWidgetItem(f"{p.name} (EAR Threshold: {p.ear_threshold:.2f}, MAR: {p.mar_threshold:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.profile_list.addItem(item)

        if not self.profiles:
            self.clear_details()

    def clear_details(self):
        """Reset text labels."""
        self.lbl_name.setText("None")
        self.lbl_ear.setText("-")
        self.lbl_mar.setText("-")
        self.lbl_blink.setText("-")
        self.load_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

    def on_profile_selected(self, current, previous):
        """Update detail readouts when a profile item is clicked."""
        if not current:
            self.clear_details()
            return
        p: UserProfile = current.data(Qt.ItemDataRole.UserRole)
        self.lbl_name.setText(p.name)
        self.lbl_ear.setText(f"{p.baseline_ear:.3f}  -->  Alert at {p.ear_threshold:.3f}")
        self.lbl_mar.setText(f"{p.baseline_mar:.3f}  -->  Yawn at {p.mar_threshold:.3f}")
        self.lbl_blink.setText(f"{p.blink_rate:.1f} blinks/min")
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

    def save_current_profile(self):
        """Persist current active thresholds as a new profile."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a profile name.")
            return

        thr = self.config.get("thresholds", {})
        ear_thr = thr.get("ear_threshold", 0.21)
        mar_thr = thr.get("mar_threshold", 0.60)
        baseline_ear = round(ear_thr / 0.70, 3)  # Approximate baseline from threshold
        baseline_mar = round(mar_thr / 1.50, 3)

        profile = UserProfile.save_or_update(
            name=name,
            baseline_ear=baseline_ear,
            ear_threshold=ear_thr,
            baseline_mar=baseline_mar,
            mar_threshold=mar_thr,
            blink_rate=15.0
        )
        QMessageBox.information(self, "Saved", f"Profile '{name}' saved successfully!")
        self.name_input.clear()
        self.load_profiles_list()

    def delete_selected_profile(self):
        """Delete highlighted profile."""
        item = self.profile_list.currentItem()
        if not item:
            return
        p: UserProfile = item.data(Qt.ItemDataRole.UserRole)
        confirm = QMessageBox.question(
            self, "Confirm Delete", f"Delete profile '{p.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes and p.id is not None:
            UserProfile.delete(p.id)
            self.load_profiles_list()

    def apply_profile(self):
        """Emit selected profile and close dialog."""
        item = self.profile_list.currentItem()
        if not item:
            return
        p: UserProfile = item.data(Qt.ItemDataRole.UserRole)
        self.profile_loaded.emit(p)
        self.accept()
