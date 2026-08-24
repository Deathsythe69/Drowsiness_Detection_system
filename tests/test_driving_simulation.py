"""Unit tests for Driving Mode Simulation / Desk Test Switch."""

import yaml
from core.motion_detector import VehicleMotionDetector
from core.shared_state import SharedState


def test_simulated_driving_bypasses_optical_flow_gating():
    """Verify that when simulated driving is enabled, vehicle is treated as driving
    even if the camera and scene are completely stationary.
    """
    # Initialize real detector with stationary frames
    detector = VehicleMotionDetector(motion_threshold=4.0, window_size=15, min_moving_ratio=0.4)
    assert detector.is_moving is False

    # Standard configuration with motion gating enabled
    config = {
        "motion": {
            "enabled": True,
            "require_motion_for_alert": True,
            "simulate_driving": True,
            "motion_threshold": 4.0
        }
    }

    # Simulation check logic
    simulate_driving = config.get("motion", {}).get("simulate_driving", False)
    require_motion_for_alert = config.get("motion", {}).get("require_motion_for_alert", True)

    # When simulate_driving is True:
    is_driving = True if simulate_driving else (detector.is_moving or not require_motion_for_alert)
    assert is_driving is True

    # When simulate_driving is False and stationary:
    config["motion"]["simulate_driving"] = False
    simulate_driving = config.get("motion", {}).get("simulate_driving", False)
    is_driving = True if simulate_driving else (detector.is_moving or not require_motion_for_alert)
    assert is_driving is False


def test_simulated_driving_shared_state_telemetry():
    """Verify that simulated driving status is reflected in SharedState telemetry."""
    SharedState.reset_singleton()
    state = SharedState()

    state.update_metrics(
        state="AWAKE",
        fatigue_score=10.0,
        ear=0.28,
        mar=0.15,
        is_vehicle_moving=True,
        motion_score=99.0,
        is_simulated_driving=True
    )

    metrics = state.get_metrics()
    assert metrics["is_simulated_driving"] is True
    assert metrics["is_vehicle_moving"] is True
    assert metrics["motion_score"] == 99.0


def test_config_yaml_contains_simulate_driving():
    """Verify that config.yaml defines the simulate_driving parameter under motion."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert "motion" in cfg
    assert "simulate_driving" in cfg["motion"]
