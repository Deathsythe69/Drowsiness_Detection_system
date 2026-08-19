"""Models Module.

Defines Session and Event models and database operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from storage.db import get_db_connection

@dataclass
class Session:
    """Represents a focus / tracking session."""
    id: Optional[int]
    user_label: str
    start_time: str
    end_time: Optional[str]
    baseline_ear: float

    @classmethod
    def create(cls, user_label: str, baseline_ear: float) -> "Session":
        """Insert a new session entry into database.
        
        Args:
            user_label: Tag indicating the user.
            baseline_ear: Calculated baseline EAR from calibration.
            
        Returns:
            Session: Newly created session object.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        start_time = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO sessions (user_label, start_time, baseline_ear) VALUES (?, ?, ?)",
            (user_label, start_time, baseline_ear)
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cls(id=session_id, user_label=user_label, start_time=start_time, end_time=None, baseline_ear=baseline_ear)

    def end(self):
        """Write session end timestamp to database."""
        if self.id is None:
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        self.end_time = datetime.now().isoformat()
        cursor.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (self.end_time, self.id)
        )
        conn.commit()
        conn.close()

@dataclass
class Event:
    """Represents a specific metric event (e.g. blink, yawn, warning, alert)."""
    id: Optional[int]
    session_id: int
    timestamp: str
    event_type: str
    ear_value: float
    mar_value: float

    @classmethod
    def log(cls, session_id: int, event_type: str, ear_value: float, mar_value: float) -> "Event":
        """Write an event log entry to the database.
        
        Args:
            session_id: Active session id.
            event_type: Alert label (drowsy_warning, drowsy_alert, yawn, blink).
            ear_value: Current average EAR.
            mar_value: Current MAR.
            
        Returns:
            Event: Newly logged event object.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO events (session_id, timestamp, event_type, ear_value, mar_value) VALUES (?, ?, ?, ?, ?)",
            (session_id, timestamp, event_type, ear_value, mar_value)
        )
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cls(id=event_id, session_id=session_id, timestamp=timestamp, event_type=event_type, ear_value=ear_value, mar_value=mar_value)


# --- Helper Database Queries for Summary Panel ---

def get_all_sessions_list() -> List[Dict[str, Any]]:
    """Retrieve all sessions ordered by most recent.
    
    Returns:
        List[Dict[str, Any]]: Session attributes list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_label, start_time, end_time, baseline_ear FROM sessions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_session_events_list(session_id: int) -> List[Dict[str, Any]]:
    """Retrieve all event details associated with a session.
    
    Args:
        session_id: ID of the target session.
        
    Returns:
        List[Dict[str, Any]]: Events list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, event_type, ear_value, mar_value FROM events WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_session(session_id: int):
    """Delete a session and all its associated events.
    
    Args:
        session_id: ID of the session to delete.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Cascades automatically if foreign keys enabled, otherwise manual cleanup:
    cursor.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
