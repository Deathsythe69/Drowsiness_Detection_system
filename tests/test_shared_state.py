"""Tests for SharedState thread-safe singleton module."""

import threading
from core.shared_state import SharedState, BuzzerCommand


class TestSharedStateSingleton:
    """Verify the singleton pattern of SharedState."""

    def setup_method(self):
        """Reset singleton before each test."""
        SharedState.reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        SharedState.reset_singleton()

    def test_singleton_returns_same_instance(self):
        """SharedState() should always return the same object."""
        a = SharedState()
        b = SharedState()
        assert a is b

    def test_reset_singleton_creates_new_instance(self):
        """After reset_singleton(), a new instance is created."""
        a = SharedState()
        SharedState.reset_singleton()
        b = SharedState()
        assert a is not b


class TestSharedStateMetrics:
    """Verify thread-safe metric updates and reads."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.state = SharedState()

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_default_metrics(self):
        """Default metrics should have reasonable initial values."""
        m = self.state.get_metrics()
        assert m["state"] == "INITIALIZING"
        assert m["fatigue_score"] == 0.0
        assert m["ear"] == 0.0
        assert m["mar"] == 0.0
        assert m["is_buzzer_playing"] is False
        assert m["is_simulated_driving"] is False

    def test_update_and_get_metrics(self):
        """Metrics should reflect the latest update."""
        self.state.update_metrics(
            state="DROWSY_ALERT",
            fatigue_score=85.3,
            ear=0.185,
            mar=0.72,
            pitch=-22.5,
            yaw=5.0,
            is_buzzer_playing=True,
            is_low_light=True,
            luminance=42.5,
            active_yawn_count=4,
            session_seconds=120,
            faces_detected=2,
            is_vehicle_moving=True,
            motion_score=8.5,
            is_simulated_driving=True,
            shoulder_angle=14.5,
            is_slouched=True,
            is_frozen_still=False,
            pose_staleness_frames=2
        )
        m = self.state.get_metrics()
        assert m["state"] == "DROWSY_ALERT"
        assert m["fatigue_score"] == 85.3
        assert m["ear"] == 0.185
        assert m["mar"] == 0.72
        assert m["pitch"] == -22.5
        assert m["is_buzzer_playing"] is True
        assert m["is_low_light"] is True
        assert m["active_yawn_count"] == 4
        assert m["session_seconds"] == 120
        assert m["faces_detected"] == 2
        assert m["is_vehicle_moving"] is True
        assert m["motion_score"] == 8.5
        assert m["is_simulated_driving"] is True
        assert m["shoulder_angle"] == 14.5
        assert m["is_slouched"] is True
        assert m["is_frozen_still"] is False
        assert m["pose_staleness_frames"] == 2

    def test_concurrent_updates(self):
        """Multiple threads writing metrics should not cause data corruption."""
        errors = []

        def writer(state_val, ear_val):
            try:
                for _ in range(100):
                    self.state.update_metrics(
                        state=state_val, fatigue_score=50.0,
                        ear=ear_val, mar=0.5
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("AWAKE", 0.30)),
            threading.Thread(target=writer, args=("DROWSY_ALERT", 0.15)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        m = self.state.get_metrics()
        assert m["state"] in ("AWAKE", "DROWSY_ALERT")


class TestBuzzerCommands:
    """Verify buzzer command queuing and consumption."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.state = SharedState()

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_no_pending_command_by_default(self):
        """No command should be pending initially."""
        cmd, ip = self.state.consume_buzzer_command()
        assert cmd == BuzzerCommand.NONE
        assert ip == ""

    def test_mute_command_queued_and_consumed(self):
        """A mute command should be retrievable once."""
        self.state.set_buzzer_command(BuzzerCommand.MUTE, "192.168.1.10")
        cmd, ip = self.state.consume_buzzer_command()
        assert cmd == BuzzerCommand.MUTE
        assert ip == "192.168.1.10"

        # After consumption, command should be cleared
        cmd2, _ = self.state.consume_buzzer_command()
        assert cmd2 == BuzzerCommand.NONE

    def test_unmute_command(self):
        """Unmute command should work identically."""
        self.state.set_buzzer_command(BuzzerCommand.UNMUTE, "10.0.0.5")
        cmd, ip = self.state.consume_buzzer_command()
        assert cmd == BuzzerCommand.UNMUTE
        assert ip == "10.0.0.5"


class TestEventBuffer:
    """Verify the recent events push/get buffer."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.state = SharedState()

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_push_and_get_events(self):
        """Pushed events should appear in get_recent_events."""
        self.state.push_event("yawn", "2026-08-19T15:00:00")
        self.state.push_event("blink", "2026-08-19T15:00:01")
        events = self.state.get_recent_events()
        assert len(events) == 2
        assert events[0]["event_type"] == "yawn"
        assert events[1]["event_type"] == "blink"

    def test_event_buffer_capped_at_50(self):
        """Buffer should not exceed 50 events."""
        for i in range(60):
            self.state.push_event(f"event_{i}", f"2026-08-19T15:00:{i:02d}")
        events = self.state.get_recent_events()
        assert len(events) == 50
        # Should keep the latest 50
        assert events[0]["event_type"] == "event_10"


class TestMonitoringState:
    """Verify monitoring active/inactive state tracking."""

    def setup_method(self):
        SharedState.reset_singleton()
        self.state = SharedState()

    def teardown_method(self):
        SharedState.reset_singleton()

    def test_default_not_monitoring(self):
        assert self.state.is_monitoring() is False

    def test_set_monitoring(self):
        self.state.set_monitoring(True)
        assert self.state.is_monitoring() is True
        self.state.set_monitoring(False)
        assert self.state.is_monitoring() is False
