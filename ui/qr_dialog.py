"""Wi-Fi & Mobile Access Dialog with Scannable QR Code.

Enables supervisors or passengers to scan a QR code from their mobile device
(over Wi-Fi or Hotspot) to immediately access the remote admin buzzer control panel.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QApplication, QMessageBox
)
from PyQt6.QtCore import Qt, QByteArray
from PyQt6.QtGui import QPixmap, QImage, QColor
import webbrowser

from core.remote_admin import generate_qr_code_bytes, get_network_info


class WiFiAccessDialog(QDialog):
    """Dialog displaying a scannable QR Code and Wi-Fi connection info."""

    def __init__(self, port: int = 8080, parent=None):
        """Initialize Wi-Fi QR Access dialog.
        
        Args:
            port: Port the remote admin server is running on.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.port = port
        self.setWindowTitle("Mobile & Wi-Fi Access — Remote Admin Panel")
        self.setFixedSize(480, 560)
        self.setObjectName("wifi_dialog")
        self.init_ui()

    def init_ui(self):
        """Build UI layout with QR code preview and network details."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Title & Subtitle
        title = QLabel("📱 Mobile & Wi-Fi Buzzer Control")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 17px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        subtitle = QLabel("Scan this QR code with your phone camera to control the buzzer over Wi-Fi:")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #a1a1aa;")
        layout.addWidget(subtitle)

        # Get network information
        net_info = get_network_info()
        primary_ip = net_info.get("primary_ip", "127.0.0.1")
        self.primary_url = f"http://{primary_ip}:{self.port}"

        # QR Code Display Frame
        qr_frame = QFrame()
        qr_frame.setStyleSheet(
            "background-color: #ffffff; border-radius: 16px; padding: 12px;"
        )
        qr_layout = QVBoxLayout(qr_frame)
        qr_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        try:
            png_bytes = generate_qr_code_bytes(self.primary_url)
            qimg = QImage.fromData(png_bytes)
            pixmap = QPixmap.fromImage(qimg).scaled(
                220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.qr_label.setPixmap(pixmap)
        except Exception as e:
            self.qr_label.setText(f"Error generating QR:\n{e}")
            self.qr_label.setStyleSheet("color: #dc2626;")

        qr_layout.addWidget(self.qr_label)
        layout.addWidget(qr_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        # Direct URL Display Card
        url_card = QFrame()
        url_card.setStyleSheet(
            "background-color: #1a1a1e; border: 1px solid #2e2e33; border-radius: 10px; padding: 10px;"
        )
        url_layout = QVBoxLayout(url_card)
        url_layout.setSpacing(4)

        url_label = QLabel(self.primary_url)
        url_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        url_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #60a5fa; letter-spacing: 0.5px;")
        url_layout.addWidget(url_label)

        # Network interface list
        interfaces = net_info.get("interfaces", {})
        if len(interfaces) > 1:
            alt_text = "Available Network Interfaces:\n" + "\n".join(
                f"• {name}: http://{ip}:{self.port}" for name, ip in interfaces.items()
            )
            alt_lbl = QLabel(alt_text)
            alt_lbl.setStyleSheet("font-size: 11px; color: #71717a;")
            url_layout.addWidget(alt_lbl)

        layout.addWidget(url_card)

        # Action Buttons Row (Copy URL, Open in Browser)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.copy_btn = QPushButton("📋 Copy URL")
        self.copy_btn.setObjectName("secondary_btn")
        self.copy_btn.clicked.connect(self.copy_url)
        btn_row.addWidget(self.copy_btn)

        self.open_browser_btn = QPushButton("🌐 Open Browser")
        self.open_browser_btn.setObjectName("secondary_btn")
        self.open_browser_btn.clicked.connect(self.open_browser)
        btn_row.addWidget(self.open_browser_btn)

        layout.addLayout(btn_row)

        # Close Button
        self.close_btn = QPushButton("Done")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

    def copy_url(self):
        """Copy the primary URL to system clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.primary_url)
        QMessageBox.information(self, "Copied", f"URL copied to clipboard:\n{self.primary_url}")

    def open_browser(self):
        """Open the admin panel in the default system web browser."""
        try:
            webbrowser.open(self.primary_url)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open browser: {e}")
