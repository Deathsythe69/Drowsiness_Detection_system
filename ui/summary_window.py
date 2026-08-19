"""Summary Window.

Embeds a Matplotlib plot to visualize session EAR metrics and events,
displays summary cards, and exports session logs to CSV.
"""

import csv
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QFrame, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from storage.models import get_all_sessions_list, get_session_events_list, delete_session

class MplCanvas(FigureCanvas):
    """Custom canvas designed to match the application's dark theme."""

    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1a1a1e')
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e24')
        
        # Style chart axis lines and tick labels
        for spine in self.axes.spines.values():
            spine.set_color('#3f3f46')
        self.axes.tick_params(colors='#a1a1aa', labelsize=9)
        self.axes.yaxis.label.set_color('#a1a1aa')
        self.axes.xaxis.label.set_color('#a1a1aa')
        self.axes.title.set_color('#ffffff')
        
        super().__init__(fig)


class SummaryWindow(QDialog):
    """Session history dashboard window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session History & Summary Dashboard")
        self.resize(850, 600)
        self.init_ui()
        self.load_sessions_dropdown()

    def init_ui(self):
        """Create components for summary dashboard layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # Title Label
        title_label = QLabel("Session History Dashboard")
        title_label.setObjectName("header")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Dropdown Selection & Control Panel
        control_layout = QHBoxLayout()
        
        self.session_selector = QComboBox()
        self.session_selector.setMinimumWidth(300)
        self.session_selector.currentIndexChanged.connect(self.on_session_changed)
        control_layout.addWidget(QLabel("Select Session:"))
        control_layout.addWidget(self.session_selector)
        
        control_layout.addStretch()

        self.export_btn = QPushButton("Export Event Logs (CSV)")
        self.export_btn.clicked.connect(self.export_csv)
        control_layout.addWidget(self.export_btn)

        self.delete_btn = QPushButton("Delete Session")
        self.delete_btn.setObjectName("danger_btn")
        self.delete_btn.clicked.connect(self.on_delete_session)
        control_layout.addWidget(self.delete_btn)
        
        main_layout.addLayout(control_layout)

        # Stats Cards Layout
        stats_layout = QHBoxLayout()
        
        # Card 1: Duration
        self.card_duration = QFrame()
        self.card_duration.setObjectName("card")
        self.card_duration.setFrameStyle(QFrame.Shape.StyledPanel)
        d_layout = QVBoxLayout(self.card_duration)
        d_layout.addWidget(QLabel("TRACKING DURATION"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_duration = QLabel("00:00")
        self.lbl_duration.setStyleSheet("font-size: 20px; font-weight: bold; color: #60a5fa;")
        d_layout.addWidget(self.lbl_duration, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.card_duration)
        
        # Card 2: Alerts count
        self.card_alerts = QFrame()
        self.card_alerts.setObjectName("card")
        self.card_alerts.setFrameStyle(QFrame.Shape.StyledPanel)
        a_layout = QVBoxLayout(self.card_alerts)
        a_layout.addWidget(QLabel("DROWSY ALERTS"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_alerts = QLabel("0")
        self.lbl_alerts.setStyleSheet("font-size: 20px; font-weight: bold; color: #ef4444;")
        a_layout.addWidget(self.lbl_alerts, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.card_alerts)
        
        # Card 3: Yawns count
        self.card_yawns = QFrame()
        self.card_yawns.setObjectName("card")
        self.card_yawns.setFrameStyle(QFrame.Shape.StyledPanel)
        y_layout = QVBoxLayout(self.card_yawns)
        y_layout.addWidget(QLabel("TOTAL YAWNS"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_yawns = QLabel("0")
        self.lbl_yawns.setStyleSheet("font-size: 20px; font-weight: bold; color: #f59e0b;")
        y_layout.addWidget(self.lbl_yawns, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.card_yawns)

        # Card 4: Blinks count
        self.card_blinks = QFrame()
        self.card_blinks.setObjectName("card")
        self.card_blinks.setFrameStyle(QFrame.Shape.StyledPanel)
        b_layout = QVBoxLayout(self.card_blinks)
        b_layout.addWidget(QLabel("TOTAL BLINKS"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_blinks = QLabel("0")
        self.lbl_blinks.setStyleSheet("font-size: 20px; font-weight: bold; color: #10b981;")
        b_layout.addWidget(self.lbl_blinks, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.card_blinks)
        
        main_layout.addLayout(stats_layout)

        # Matplotlib Graph Canvas
        self.canvas = MplCanvas(self, width=6, height=3, dpi=100)
        main_layout.addWidget(self.canvas, stretch=2)

    def load_sessions_dropdown(self):
        """Query sessions list and load combobox items."""
        self.session_selector.blockSignals(True)
        self.session_selector.clear()
        
        sessions = get_all_sessions_list()
        for s in sessions:
            # Parse and format the start time nicely
            try:
                dt = datetime.fromisoformat(s["start_time"])
                formatted_time = dt.strftime("%b %d, %Y - %H:%M:%S")
            except Exception:
                formatted_time = s["start_time"]
                
            label = f"{formatted_time} [{s['user_label']}]"
            self.session_selector.addItem(label, s)
            
        self.session_selector.blockSignals(False)
        self.on_session_changed()

    def on_session_changed(self):
        """Update metrics card and redrafts matplotlib chart when session selection changes."""
        # Clean the plot
        self.canvas.axes.clear()
        
        if self.session_selector.currentIndex() < 0:
            self.lbl_duration.setText("N/A")
            self.lbl_alerts.setText("0")
            self.lbl_yawns.setText("0")
            self.lbl_blinks.setText("0")
            self.canvas.axes.text(
                0.5, 0.5, "No Session Selected", 
                color="#ffffff", ha='center', va='center'
            )
            self.canvas.draw()
            return
            
        session_data = self.session_selector.currentData()
        session_id = session_data["id"]
        
        # Load stats
        events = get_session_events_list(session_id)
        
        alert_count = sum(1 for e in events if e["event_type"] == "drowsy_alert")
        yawn_count = sum(1 for e in events if e["event_type"] == "yawn")
        blink_count = sum(1 for e in events if e["event_type"] == "blink")
        
        # Calculate duration
        start_dt = datetime.fromisoformat(session_data["start_time"])
        if session_data["end_time"]:
            end_dt = datetime.fromisoformat(session_data["end_time"])
            duration_sec = (end_dt - start_dt).total_seconds()
        elif events:
            # Fallback to last event timestamp
            last_dt = datetime.fromisoformat(events[-1]["timestamp"])
            duration_sec = (last_dt - start_dt).total_seconds()
        else:
            duration_sec = 0
            
        min_part, sec_part = divmod(int(duration_sec), 60)
        self.lbl_duration.setText(f"{min_part:02d}:{sec_part:02d}")
        self.lbl_alerts.setText(str(alert_count))
        self.lbl_yawns.setText(str(yawn_count))
        self.lbl_blinks.setText(str(blink_count))
        
        # Draw EAR line plot
        # Parse time offsets in seconds
        times = []
        ears = []
        
        # Sort out specific events to plot as dots
        alert_times, alert_values = [], []
        yawn_times, yawn_values = [], []
        warning_times, warning_values = [], []

        for e in events:
            evt_dt = datetime.fromisoformat(e["timestamp"])
            offset = (evt_dt - start_dt).total_seconds()
            
            # Matplotlib line connects all frame events
            times.append(offset)
            ears.append(e["ear_value"])
            
            if e["event_type"] == "drowsy_alert":
                alert_times.append(offset)
                alert_values.append(e["ear_value"])
            elif e["event_type"] == "drowsy_warning":
                warning_times.append(offset)
                warning_values.append(e["ear_value"])
            elif e["event_type"] == "yawn":
                yawn_times.append(offset)
                yawn_values.append(e["ear_value"])
                
        # Draw main line
        if times:
            self.canvas.axes.plot(times, ears, color="#3b82f6", linewidth=1.5, label="EAR Metric")
            
            # Overlay event scattered dots
            if alert_times:
                self.canvas.axes.scatter(alert_times, alert_values, color="#ef4444", s=50, marker='o', label="Drowsy Alert")
            if warning_times:
                self.canvas.axes.scatter(warning_times, warning_values, color="#f59e0b", s=30, marker='s', label="Warning")
            if yawn_times:
                self.canvas.axes.scatter(yawn_times, yawn_values, color="#d946ef", s=40, marker='^', label="Yawn Event")
                
            # Draw baseline horizontal line
            baseline = session_data["baseline_ear"]
            if baseline > 0:
                self.canvas.axes.axhline(baseline, color="#10b981", linestyle="--", alpha=0.6, label="Calibrated Baseline")
                
            self.canvas.axes.set_xlabel("Session Duration (Seconds)")
            self.canvas.axes.set_ylabel("Eye Aspect Ratio (EAR)")
            self.canvas.axes.set_title(f"Fatigue Metrics Profile: Session #{session_id}")
            self.canvas.axes.legend(
                facecolor='#1a1a1e', edgecolor='#3f3f46', 
                labelcolor='#ffffff', loc='upper right', fontsize=8
            )
            self.canvas.axes.grid(True, color='#27272a', linestyle=':', alpha=0.5)
        else:
            self.canvas.axes.text(
                0.5, 0.5, "No tracking metrics logged during this session.", 
                color="#a1a1aa", ha='center', va='center'
            )
            
        self.canvas.draw()

    def export_csv(self):
        """Export session events to a CSV file."""
        if self.session_selector.currentIndex() < 0:
            return
            
        session_data = self.session_selector.currentData()
        session_id = session_data["id"]
        
        events = get_session_events_list(session_id)
        if not events:
            QMessageBox.information(self, "No Logs", "No events logged for this session.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session Log", f"session_{session_id}_log.csv", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Timestamp", "Event Type", "EAR Value", "MAR Value"])
                for e in events:
                    writer.writerow([
                        e["id"],
                        e["timestamp"],
                        e["event_type"],
                        e["ear_value"],
                        e["mar_value"]
                    ])
            QMessageBox.information(self, "Success", "Logs exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export log: {e}")

    def on_delete_session(self):
        """Confirm and delete selected session."""
        if self.session_selector.currentIndex() < 0:
            return
            
        session_data = self.session_selector.currentData()
        session_id = session_data["id"]
        
        reply = QMessageBox.question(
            self, "Delete Session",
            f"Are you sure you want to delete Session #{session_id}? This action is permanent.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                delete_session(session_id)
                self.load_sessions_dropdown()
            except Exception as e:
                QMessageBox.critical(self, "Deletion Error", f"Failed to delete session: {e}")
