"""Remote Admin Web Panel Module.

Provides a lightweight Flask-based web server running in a daemon thread
that allows supervisors to monitor system status and control the buzzer
from any device on the same LAN via a web browser.

All remote override actions are PIN-authenticated and audit-logged.
"""

import os
import io
import socket
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List
import psutil
import qrcode

from flask import (
    Flask, request, redirect, url_for, session,
    jsonify, render_template_string, Response, send_from_directory
)

from core.shared_state import SharedState, BuzzerCommand

# Suppress Flask/Werkzeug request logging noise in the PyQt console
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# HTML Templates (embedded to avoid external template file dependencies)
# ---------------------------------------------------------------------------

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login — Drowsiness Monitor</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', -apple-system, Roboto, 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #0f0f11 0%, #1a1a2e 50%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            background: rgba(24, 24, 27, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(63, 63, 70, 0.6);
            border-radius: 20px;
            padding: 40px 36px;
            width: 380px;
            max-width: 95vw;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
        }
        .login-card h1 {
            font-size: 20px;
            text-align: center;
            margin-bottom: 6px;
            color: #fff;
        }
        .login-card .subtitle {
            text-align: center;
            font-size: 13px;
            color: #a1a1aa;
            margin-bottom: 28px;
        }
        .login-card .icon {
            text-align: center;
            font-size: 42px;
            margin-bottom: 16px;
        }
        label {
            display: block;
            font-size: 12px;
            color: #a1a1aa;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        input[type="password"] {
            width: 100%;
            padding: 12px 16px;
            background: #121214;
            border: 1px solid #3f3f46;
            border-radius: 10px;
            color: #fff;
            font-size: 18px;
            letter-spacing: 6px;
            text-align: center;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="password"]:focus {
            border-color: #f59e0b;
            box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15);
        }
        button {
            width: 100%;
            margin-top: 20px;
            padding: 12px;
            background: linear-gradient(135deg, #d97706, #b45309);
            color: #fff;
            font-size: 15px;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(217, 119, 6, 0.35);
        }
        button:active { transform: translateY(0); }
        .error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            color: #fca5a5;
            font-size: 13px;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <div class="icon">🔐</div>
        <h1>Admin Remote Control</h1>
        <p class="subtitle">Drowsiness & Attention Monitoring System</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/auth">
            <label for="pin">Administrator PIN</label>
            <input type="password" id="pin" name="pin" placeholder="••••" autocomplete="off" autofocus required>
            <button type="submit">Authenticate & Access</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Remote Admin Dashboard — Drowsiness Monitor</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', -apple-system, Roboto, 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #0f0f11 0%, #1a1a2e 50%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 16px 24px;
            background: rgba(24, 24, 27, 0.8);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(63, 63, 70, 0.5);
            border-radius: 16px;
        }
        .header h1 {
            font-size: 18px;
            color: #fff;
        }
        .header .badge {
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .badge-awake { background: #059669; color: #fff; }
        .badge-warning { background: #d97706; color: #fff; }
        .badge-alert { background: #dc2626; color: #fff; animation: pulse 1s infinite; }
        .badge-noface { background: #4b5563; color: #fff; }
        .badge-init { background: #3f3f46; color: #a1a1aa; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(24, 24, 27, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(63, 63, 70, 0.4);
            border-radius: 16px;
            padding: 20px;
        }
        .card h3 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #71717a;
            margin-bottom: 12px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(63, 63, 70, 0.3);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { font-size: 13px; color: #a1a1aa; }
        .metric-value { font-size: 15px; font-weight: 700; color: #fff; }

        .fatigue-bar {
            width: 100%;
            height: 8px;
            background: #27272a;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .fatigue-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease, background 0.3s;
        }

        .buzzer-controls {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }
        .btn {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.2s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .btn-mute {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            color: #fff;
            box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
        }
        .btn-unmute {
            background: linear-gradient(135deg, #059669, #047857);
            color: #fff;
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
        }
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }

        .buzzer-status {
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .buzzer-on {
            background: rgba(220, 38, 38, 0.2);
            border: 1px solid #dc2626;
            color: #fca5a5;
            animation: pulse 1s infinite;
        }
        .buzzer-off {
            background: rgba(5, 150, 105, 0.15);
            border: 1px solid #059669;
            color: #6ee7b7;
        }

        .event-log {
            max-height: 260px;
            overflow-y: auto;
            font-size: 12px;
        }
        .event-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(63, 63, 70, 0.2);
        }
        .event-type { color: #60a5fa; font-weight: 600; }
        .event-time { color: #71717a; font-size: 11px; }

        .logout-btn {
            display: inline-block;
            padding: 6px 16px;
            background: #3f3f46;
            color: #a1a1aa;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            text-decoration: none;
        }
        .logout-btn:hover { background: #52525b; color: #fff; }

        .session-timer {
            font-size: 20px;
            font-weight: 700;
            color: #fff;
            text-align: center;
            margin-bottom: 8px;
        }

        @media (max-width: 600px) {
            .grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 10px; text-align: center; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ Remote Admin Dashboard</h1>
        <div>
            <span class="badge badge-init" id="state-badge">LOADING...</span>
            <a href="/logout" class="logout-btn" style="margin-left:10px;">Logout</a>
        </div>
    </div>

    <div class="grid">
        <!-- Metrics Card -->
        <div class="card">
            <h3>Real-Time Metrics</h3>
            <div class="session-timer" id="session-timer">00:00</div>
            <div class="metric">
                <span class="metric-label">Eye Aspect Ratio (EAR)</span>
                <span class="metric-value" id="ear">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Mouth Aspect Ratio (MAR)</span>
                <span class="metric-value" id="mar">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Head Pitch</span>
                <span class="metric-value" id="pitch">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Head Yaw</span>
                <span class="metric-value" id="yaw">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Rolling Yawns (5m)</span>
                <span class="metric-value" id="yawns">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Faces Detected</span>
                <span class="metric-value" id="faces">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Low-Light Mode</span>
                <span class="metric-value" id="lowlight">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Vehicle Movement</span>
                <span class="metric-value" id="vehicle-motion">—</span>
            </div>
            <h3 style="margin-top:12px;">Fatigue Score</h3>
            <div class="fatigue-bar">
                <div class="fatigue-fill" id="fatigue-fill" style="width:0%; background:#059669;"></div>
            </div>
            <div style="text-align:right; font-size:12px; color:#71717a; margin-top:4px;">
                <span id="fatigue-val">0</span> / 100
            </div>
        </div>

        <!-- Buzzer Control Card -->
        <div class="card">
            <h3>Buzzer Control</h3>
            <div class="buzzer-status buzzer-off" id="buzzer-status">
                🔇 BUZZER SILENT
            </div>
            <div class="buzzer-controls">
                <button class="btn btn-mute" id="mute-btn" onclick="sendCommand('mute')">
                    🔇 Mute Alarm
                </button>
                <button class="btn btn-unmute" id="unmute-btn" onclick="sendCommand('unmute')">
                    🔊 Re-Enable
                </button>
            </div>

            <h3 style="margin-top:24px;">Recent Event Log</h3>
            <div class="event-log" id="event-log">
                <p style="color:#71717a; text-align:center; padding:20px;">Loading events...</p>
            </div>
        </div>
    </div>

    <script>
        const STATE_CLASSES = {
            'AWAKE': 'badge-awake',
            'DROWSY_WARNING': 'badge-warning',
            'DROWSY_ALERT': 'badge-alert',
            'NO_FACE': 'badge-noface',
            'INITIALIZING': 'badge-init',
            'MONITOR INACTIVE': 'badge-init'
        };

        function formatDuration(seconds) {
            const m = Math.floor(seconds / 60);
            const s = seconds % 60;
            return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
        }

        function getFatigueColor(score) {
            if (score >= 75) return '#dc2626';
            if (score >= 40) return '#d97706';
            return '#059669';
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();

                // State badge
                const badge = document.getElementById('state-badge');
                badge.textContent = data.state;
                badge.className = 'badge ' + (STATE_CLASSES[data.state] || 'badge-init');

                // Metrics
                document.getElementById('ear').textContent = data.ear.toFixed(3);
                document.getElementById('mar').textContent = data.mar.toFixed(3);
                document.getElementById('pitch').textContent = data.pitch.toFixed(1) + '°';
                document.getElementById('yaw').textContent = data.yaw.toFixed(1) + '°';
                document.getElementById('yawns').textContent = data.active_yawn_count;
                document.getElementById('faces').textContent = data.faces_detected;
                document.getElementById('lowlight').textContent = data.is_low_light ? '🌙 Active' : '☀️ Normal';
                document.getElementById('vehicle-motion').textContent = data.is_simulated_driving ? '🚗 Driving (Test Mode)' : (data.is_vehicle_moving ? '🚗 Driving (Active)' : '🛑 Parked / Stopped');
                document.getElementById('session-timer').textContent = formatDuration(data.session_seconds);

                // Fatigue bar
                const score = data.fatigue_score;
                document.getElementById('fatigue-fill').style.width = score + '%';
                document.getElementById('fatigue-fill').style.background = getFatigueColor(score);
                document.getElementById('fatigue-val').textContent = score.toFixed(1);

                // Buzzer status
                const buzzerEl = document.getElementById('buzzer-status');
                if (data.is_buzzer_playing) {
                    buzzerEl.className = 'buzzer-status buzzer-on';
                    buzzerEl.innerHTML = '🔊 BUZZER ACTIVE';
                } else {
                    buzzerEl.className = 'buzzer-status buzzer-off';
                    buzzerEl.innerHTML = '🔇 BUZZER SILENT';
                }
            } catch (e) {
                console.error('Status fetch error:', e);
            }
        }

        async function fetchEvents() {
            try {
                const res = await fetch('/log');
                const events = await res.json();
                const logEl = document.getElementById('event-log');

                if (events.length === 0) {
                    logEl.innerHTML = '<p style="color:#71717a;text-align:center;padding:20px;">No events yet.</p>';
                    return;
                }

                let html = '';
                for (const evt of events.reverse().slice(0, 30)) {
                    const time = evt.timestamp ? evt.timestamp.split('T')[1]?.substring(0, 8) || '' : '';
                    html += '<div class="event-row">' +
                        '<span class="event-type">' + evt.event_type + '</span>' +
                        '<span class="event-time">' + time + '</span>' +
                        '</div>';
                }
                logEl.innerHTML = html;
            } catch (e) {
                console.error('Event log fetch error:', e);
            }
        }

        async function sendCommand(cmd) {
            try {
                const res = await fetch('/buzzer/' + cmd, { method: 'POST' });
                const data = await res.json();
                if (!data.success) {
                    alert(data.message || 'Command failed');
                }
                fetchStatus();
            } catch (e) {
                alert('Network error: ' + e.message);
            }
        }

        // Poll every 1 second for live metrics
        setInterval(fetchStatus, 1000);
        setInterval(fetchEvents, 3000);

        // Initial fetch
        fetchStatus();
        fetchEvents();
    </script>
</body>
</html>
"""


def generate_qr_code_bytes(url: str) -> bytes:
    """Generate PNG image bytes for a QR code encoding the provided URL.
    
    Args:
        url: The web URL to encode in the QR code.
        
    Returns:
        bytes: PNG-encoded image byte stream.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_network_info() -> Dict[str, Any]:
    """Detect all active network interfaces (Wi-Fi, Ethernet, Mobile Hotspot).
    
    Returns:
        Dict with:
            - 'primary_ip': Best accessible IPv4 address
            - 'interfaces': Dict of interface_name -> ip_address
            - 'is_wifi': Boolean whether a dedicated Wi-Fi adapter is primary
    """
    interfaces: Dict[str, str] = {}
    try:
        for name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        interfaces[name] = ip
    except Exception:
        pass

    primary_ip = "127.0.0.1"
    is_wifi = False

    # Check for Wi-Fi interface first
    for name, ip in interfaces.items():
        if any(w in name.lower() for w in ["wi-fi", "wireless", "wlan"]):
            primary_ip = ip
            is_wifi = True
            break

    if primary_ip == "127.0.0.1" and interfaces:
        # Pick first available non-loopback IP (e.g. Ethernet / Hotspot)
        primary_ip = next(iter(interfaces.values()))

    # Fallback to UDP socket query
    if primary_ip == "127.0.0.1":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
            s.close()
        except Exception:
            primary_ip = "127.0.0.1"

    return {
        "primary_ip": primary_ip,
        "interfaces": interfaces,
        "is_wifi": is_wifi
    }


def get_local_ip() -> str:
    """Detect the machine's best LAN / Wi-Fi IP address for display.
    
    Returns:
        Local IP address string (e.g., '192.168.1.6'), or '127.0.0.1' fallback.
    """
    return get_network_info()["primary_ip"]


def create_admin_app(admin_pin: str) -> Flask:
    """Create and configure the Flask admin application.
    
    Args:
        admin_pin: Required PIN for authentication.
        
    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    app.config["ADMIN_PIN"] = str(admin_pin)

    @app.route("/")
    def index():
        """Login page or redirect to dashboard if already authenticated."""
        if session.get("admin_authenticated"):
            return redirect(url_for("dashboard"))
        return render_template_string(LOGIN_HTML, error=None)

    @app.route("/auth", methods=["POST"])
    def auth():
        """Validate admin PIN and create session."""
        pin = request.form.get("pin", "").strip()
        if pin == app.config["ADMIN_PIN"]:
            session["admin_authenticated"] = True
            return redirect(url_for("dashboard"))
        return render_template_string(LOGIN_HTML, error="Incorrect PIN. Access denied.")

    @app.route("/dashboard")
    def dashboard():
        """Render the admin dashboard (requires authentication)."""
        if not session.get("admin_authenticated"):
            return redirect(url_for("index"))
        return render_template_string(DASHBOARD_HTML)

    @app.route("/status")
    def status():
        """Return current system metrics as JSON."""
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        shared = SharedState()
        return jsonify(shared.get_metrics())

    @app.route("/log")
    def event_log():
        """Return recent event log as JSON."""
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        shared = SharedState()
        return jsonify(shared.get_recent_events())

    @app.route("/qr")
    def qr_code():
        """Serve dynamic PNG QR Code for quick mobile Wi-Fi connection."""
        shared = SharedState()
        url = shared.get_remote_admin_url() or f"http://{get_local_ip()}:8080"
        png_data = generate_qr_code_bytes(url)
        return Response(png_data, mimetype="image/png")

    @app.route("/buzzer/mute", methods=["POST"])
    def mute_buzzer():
        """Mute the active alarm remotely."""
        if not session.get("admin_authenticated"):
            return jsonify({"success": False, "message": "Not authenticated"}), 401
        remote_ip = request.remote_addr or "unknown"
        shared = SharedState()
        shared.set_buzzer_command(BuzzerCommand.MUTE, remote_ip)
        return jsonify({"success": True, "message": "Mute command sent", "remote_ip": remote_ip})

    @app.route("/buzzer/unmute", methods=["POST"])
    def unmute_buzzer():
        """Re-enable alarm sound remotely."""
        if not session.get("admin_authenticated"):
            return jsonify({"success": False, "message": "Not authenticated"}), 401
        remote_ip = request.remote_addr or "unknown"
        shared = SharedState()
        shared.set_buzzer_command(BuzzerCommand.UNMUTE, remote_ip)
        return jsonify({"success": True, "message": "Unmute command sent", "remote_ip": remote_ip})

    @app.route("/logout")
    def logout():
        """Clear admin session and redirect to login."""
        session.clear()
        return redirect(url_for("index"))

    @app.route("/evidence/<path:filename>")
    def download_evidence(filename: str):
        """Serve evidence video files securely to authenticated supervisors."""
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        evidence_dir = os.path.abspath("evidence")
        return send_from_directory(evidence_dir, filename, as_attachment=False)

    @app.route("/evidence_list")
    def list_evidence():
        """Return list of recorded evidence video clips."""
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        evidence_dir = "evidence"
        videos = []
        if os.path.exists(evidence_dir):
            for fname in sorted(os.listdir(evidence_dir), reverse=True):
                if fname.endswith(".mp4") or fname.endswith(".avi"):
                    fpath = os.path.join(evidence_dir, fname)
                    stat = os.stat(fpath)
                    videos.append({
                        "filename": fname,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "url": f"/evidence/{fname}"
                    })
        return jsonify(videos)

    # -----------------------------------------------------------------------
    # Continuous Learning & Tester Fleet Feedback Endpoints
    # -----------------------------------------------------------------------

    @app.route("/api/train/status")
    def train_status():
        """Return dataset stats and continuous trainer status."""
        from core.live_trainer import LiveSampleCollector, LiveAutoTrainer
        collector = LiveSampleCollector()
        stats = collector.get_dataset_statistics()
        return jsonify({
            "status": "idle",
            "dataset": stats
        })

    @app.route("/api/train/trigger", methods=["POST"])
    def train_trigger():
        """Trigger continuous training asynchronously on accumulated tester samples."""
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        from core.live_trainer import LiveAutoTrainer
        trainer = LiveAutoTrainer()
        if trainer.is_training():
            return jsonify({"status": "already_running"}), 409
        
        trainer.train_async(epochs=4)
        return jsonify({"status": "started", "message": "Continuous self-training started in background"})

    @app.route("/api/feedback/submit_sample", methods=["POST"])
    def submit_sample():
        """Allow remote testers to submit daytime/nighttime labeled eye sample images."""
        import base64
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image_base64")
        if not img_b64:
            return jsonify({"error": "Missing image_base64"}), 400

        try:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Invalid image format"}), 400

            label = int(data.get("label", 0))
            is_low_light = bool(data.get("is_low_light", False))
            category = data.get("category", "general")
            telemetry = data.get("telemetry", {})

            from core.live_trainer import LiveSampleCollector
            collector = LiveSampleCollector()
            saved_path = collector.save_sample(
                eye_crop=img,
                full_frame=None,
                label=label,
                is_low_light=is_low_light,
                metadata=telemetry,
                category=category,
                force=True
            )
            return jsonify({"status": "saved", "path": saved_path})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def start_remote_admin_server(admin_pin: str, port: int = 8080):
    """Start the Flask admin server in a daemon background thread.
    
    The server binds to 0.0.0.0 so it's accessible from any device
    on the same Wi-Fi, Ethernet, or mobile hotspot network.
    
    Args:
        admin_pin: The admin PIN for authentication.
        port: TCP port to bind to (default 8080).
        
    Returns:
        str: The full URL where the admin panel is accessible.
    """
    app = create_admin_app(admin_pin)
    net_info = get_network_info()
    primary_ip = net_info["primary_ip"]
    url = f"http://{primary_ip}:{port}"

    shared = SharedState()
    shared.set_remote_admin_url(url)

    def run_server():
        """Run Flask in a background thread with threading enabled."""
        try:
            app.run(
                host="0.0.0.0",
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            print(f"Remote admin server error: {e}")

    server_thread = threading.Thread(target=run_server, daemon=True, name="RemoteAdminServer")
    server_thread.start()

    conn_type = "Wi-Fi" if net_info["is_wifi"] else "LAN / Network"
    print(f"[Remote Admin] Web panel started on {conn_type} at {url}")
    return url
