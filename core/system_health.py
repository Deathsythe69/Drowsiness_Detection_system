"""System Health Module.

Monitors device battery status, power source availability,
and handles camera disconnect resilience.
"""

from typing import Dict, Any, Optional
import psutil

def get_battery_status(low_threshold: int = 20) -> Dict[str, Any]:
    """Query laptop / device battery status.
    
    Args:
        low_threshold: Percentage below which battery is flagged as critical.
        
    Returns:
        Dict with battery details:
        {
            'has_battery': bool,
            'percent': int,
            'power_plugged': bool,
            'is_low': bool,
            'status_text': str
        }
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return {
                "has_battery": False,
                "percent": 100,
                "power_plugged": True,
                "is_low": False,
                "status_text": "AC Power"
            }
            
        percent = int(battery.percent)
        plugged = bool(battery.power_plugged)
        is_low = (percent <= low_threshold) and not plugged
        
        status_text = f"Battery: {percent}%" + (" (Charging)" if plugged else "")
        if is_low:
            status_text += " [LOW BATTERY]"
            
        return {
            "has_battery": True,
            "percent": percent,
            "power_plugged": plugged,
            "is_low": is_low,
            "status_text": status_text
        }
    except Exception:
        return {
            "has_battery": False,
            "percent": 100,
            "power_plugged": True,
            "is_low": False,
            "status_text": "AC Power"
        }
