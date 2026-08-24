"""Tests for Remote Admin Flask web panel."""

import json
from core.remote_admin import create_admin_app
from core.shared_state import SharedState, BuzzerCommand


class TestRemoteAdminAuth:
    """Verify PIN authentication flow."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.app = create_admin_app("9999")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_login_page_loads(self):
        """GET / should return the login page."""
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"Admin Remote Control" in resp.data

    def test_invalid_pin_denied(self):
        """Wrong PIN should show error and not grant access."""
        resp = self.client.post("/auth", data={"pin": "0000"})
        assert resp.status_code == 200
        assert b"Incorrect PIN" in resp.data

    def test_valid_pin_redirects_to_dashboard(self):
        """Correct PIN should redirect to /dashboard."""
        resp = self.client.post("/auth", data={"pin": "9999"}, follow_redirects=False)
        assert resp.status_code == 302
        assert "/dashboard" in resp.headers.get("Location", "")

    def test_dashboard_requires_auth(self):
        """GET /dashboard without auth should redirect to login."""
        resp = self.client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302

    def test_dashboard_accessible_after_auth(self):
        """After PIN auth, /dashboard should load."""
        self.client.post("/auth", data={"pin": "9999"})
        resp = self.client.get("/dashboard")
        assert resp.status_code == 200
        assert b"Remote Admin Dashboard" in resp.data

    def test_logout_clears_session(self):
        """After logout, /dashboard should redirect to login."""
        self.client.post("/auth", data={"pin": "9999"})
        self.client.get("/logout")
        resp = self.client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302


class TestRemoteAdminStatus:
    """Verify /status JSON endpoint."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.app = create_admin_app("1234")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.client.post("/auth", data={"pin": "1234"})

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_status_returns_json(self):
        """GET /status should return valid JSON with expected keys."""
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "state" in data
        assert "fatigue_score" in data
        assert "ear" in data
        assert "mar" in data
        assert "is_buzzer_playing" in data

    def test_status_reflects_shared_state(self):
        """Status should reflect values pushed to SharedState."""
        shared = SharedState()
        shared.update_metrics(
            state="DROWSY_WARNING",
            fatigue_score=55.5,
            ear=0.19,
            mar=0.65,
            pitch=-10.0,
            yaw=3.0,
            is_buzzer_playing=True,
            session_seconds=42
        )
        resp = self.client.get("/status")
        data = json.loads(resp.data)
        assert data["state"] == "DROWSY_WARNING"
        assert data["fatigue_score"] == 55.5
        assert data["ear"] == 0.19
        assert data["is_buzzer_playing"] is True
        assert data["session_seconds"] == 42

    def test_status_unauthenticated_returns_401(self):
        """Without auth, /status should return 401."""
        fresh_client = self.app.test_client()
        resp = fresh_client.get("/status")
        assert resp.status_code == 401


class TestRemoteAdminBuzzer:
    """Verify buzzer mute/unmute endpoints."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.app = create_admin_app("4321")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.client.post("/auth", data={"pin": "4321"})

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_mute_sets_shared_state_command(self):
        """POST /buzzer/mute should queue a MUTE command."""
        resp = self.client.post("/buzzer/mute")
        data = json.loads(resp.data)
        assert data["success"] is True

        shared = SharedState()
        cmd, _ = shared.consume_buzzer_command()
        assert cmd == BuzzerCommand.MUTE

    def test_unmute_sets_shared_state_command(self):
        """POST /buzzer/unmute should queue an UNMUTE command."""
        resp = self.client.post("/buzzer/unmute")
        data = json.loads(resp.data)
        assert data["success"] is True

        shared = SharedState()
        cmd, _ = shared.consume_buzzer_command()
        assert cmd == BuzzerCommand.UNMUTE

    def test_buzzer_unauthenticated_returns_401(self):
        """Without auth, buzzer endpoints should return 401."""
        fresh_client = self.app.test_client()
        resp = fresh_client.post("/buzzer/mute")
        assert resp.status_code == 401


class TestRemoteAdminEventLog:
    """Verify /log JSON endpoint."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.app = create_admin_app("5555")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.client.post("/auth", data={"pin": "5555"})

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_log_returns_empty_initially(self):
        """GET /log should return an empty list initially."""
        resp = self.client.get("/log")
        data = json.loads(resp.data)
        assert isinstance(data, list)
        assert len(data) == 0

    def test_log_returns_pushed_events(self):
        """GET /log should return events pushed to SharedState."""
        shared = SharedState()
        shared.push_event("yawn", "2026-08-19T15:00:00")
        shared.push_event("drowsy_alert", "2026-08-19T15:00:05")

        resp = self.client.get("/log")
        data = json.loads(resp.data)
        assert len(data) == 2
        assert data[0]["event_type"] == "yawn"
        assert data[1]["event_type"] == "drowsy_alert"


class TestWiFiAccessAndQRCode:
    """Verify Wi-Fi network detection and QR code generation."""

    def setup_method(self):
        self.app = create_admin_app("1234")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_qr_endpoint_returns_png(self):
        """GET /qr should return a valid PNG image stream with 200 OK."""
        resp = self.client.get("/qr")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert len(resp.data) > 100
        # Check PNG magic header bytes
        assert resp.data.startswith(b"\x89PNG\r\n\x1a\n")

    def test_generate_qr_code_bytes(self):
        """generate_qr_code_bytes should produce valid PNG bytes."""
        from core.remote_admin import generate_qr_code_bytes
        png_bytes = generate_qr_code_bytes("http://192.168.1.50:8080")
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_get_network_info_structure(self):
        """get_network_info should return primary_ip, interfaces dict, and is_wifi."""
        from core.remote_admin import get_network_info
        info = get_network_info()
        assert "primary_ip" in info
        assert "interfaces" in info
        assert "is_wifi" in info
        assert isinstance(info["interfaces"], dict)
        assert isinstance(info["is_wifi"], bool)
        assert isinstance(info["primary_ip"], str)

