"""Models Module.

Defines Session, Event, and UserProfile models with persistent database operations.
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
    """Represents a specific metric event (e.g. blink, yawn, warning, alert, admin_override)."""
    id: Optional[int]
    session_id: int
    timestamp: str
    event_type: str
    ear_value: float
    mar_value: float
    metadata: str = ""

    @classmethod
    def log(
        cls,
        session_id: int,
        event_type: str,
        ear_value: float,
        mar_value: float,
        metadata: str = ""
    ) -> "Event":
        """Write an event log entry to the database.
        
        Args:
            session_id: Active session id.
            event_type: Alert label (drowsy_warning, drowsy_alert, yawn, blink, admin_override, etc.).
            ear_value: Current average EAR.
            mar_value: Current MAR.
            metadata: Contextual details (e.g. 'Admin PIN Authenticated').
            
        Returns:
            Event: Newly logged event object.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO events (session_id, timestamp, event_type, ear_value, mar_value, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, timestamp, event_type, ear_value, mar_value, metadata)
        )
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cls(
            id=event_id,
            session_id=session_id,
            timestamp=timestamp,
            event_type=event_type,
            ear_value=ear_value,
            mar_value=mar_value,
            metadata=metadata
        )

    @classmethod
    def get_by_session(cls, session_id: int) -> List["Event"]:
        """Retrieve all events logged for a specific session.
        
        Args:
            session_id: Target session ID.
            
        Returns:
            List[Event]: Matching event records ordered chronologically.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, session_id, timestamp, event_type, ear_value, mar_value, metadata FROM events WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            cls(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                event_type=row["event_type"],
                ear_value=row["ear_value"],
                mar_value=row["mar_value"],
                metadata=row["metadata"]
            )
            for row in rows
        ]

@dataclass
class UserProfile:
    """Represents a stored user calibration profile."""
    id: Optional[int]
    name: str
    baseline_ear: float
    ear_threshold: float
    baseline_mar: float
    mar_threshold: float
    blink_rate: float
    created_at: str
    updated_at: str

    @classmethod
    def save_or_update(
        cls,
        name: str,
        baseline_ear: float,
        ear_threshold: float,
        baseline_mar: float = 0.20,
        mar_threshold: float = 0.60,
        blink_rate: float = 15.0
    ) -> "UserProfile":
        """Insert or update a named calibration profile in the database.
        
        Args:
            name: Profile identifier (e.g. 'Debasis_Normal', 'Debasis_Glasses').
            baseline_ear: Mean EAR observed during open-eye calibration.
            ear_threshold: Derived personalized closure threshold.
            baseline_mar: Baseline mouth ratio.
            mar_threshold: Derived yawn threshold.
            blink_rate: Average blinks per minute recorded.
            
        Returns:
            UserProfile: Persistent profile instance.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("SELECT id, created_at FROM user_profiles WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            profile_id = row["id"]
            created_at = row["created_at"]
            cursor.execute("""
                UPDATE user_profiles
                SET baseline_ear = ?, ear_threshold = ?, baseline_mar = ?, mar_threshold = ?, blink_rate = ?, updated_at = ?
                WHERE id = ?
            """, (baseline_ear, ear_threshold, baseline_mar, mar_threshold, blink_rate, now, profile_id))
        else:
            created_at = now
            cursor.execute("""
                INSERT INTO user_profiles (name, baseline_ear, ear_threshold, baseline_mar, mar_threshold, blink_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, baseline_ear, ear_threshold, baseline_mar, mar_threshold, blink_rate, created_at, now))
            profile_id = cursor.lastrowid
            
        conn.commit()
        conn.close()
        return cls(
            id=profile_id,
            name=name,
            baseline_ear=baseline_ear,
            ear_threshold=ear_threshold,
            baseline_mar=baseline_mar,
            mar_threshold=mar_threshold,
            blink_rate=blink_rate,
            created_at=created_at,
            updated_at=now
        )

    @classmethod
    def get_all(cls) -> List["UserProfile"]:
        """Retrieve all stored user profiles.
        
        Returns:
            List[UserProfile]: All saved profiles.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profiles ORDER BY name ASC")
        rows = cursor.fetchall()
        conn.close()
        return [
            cls(
                id=r["id"],
                name=r["name"],
                baseline_ear=r["baseline_ear"],
                ear_threshold=r["ear_threshold"],
                baseline_mar=r["baseline_mar"],
                mar_threshold=r["mar_threshold"],
                blink_rate=r["blink_rate"],
                created_at=r["created_at"],
                updated_at=r["updated_at"]
            )
            for r in rows
        ]

    @classmethod
    def delete(cls, profile_id: int):
        """Delete a profile by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_profiles WHERE id = ?", (profile_id,))
        conn.commit()
        conn.close()

# --- Helper Database Queries for Summary & Analytics ---

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
        "SELECT id, timestamp, event_type, ear_value, mar_value, metadata FROM events WHERE session_id = ? ORDER BY id ASC",
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
    cursor.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def get_all_evidence_events() -> List[Dict[str, Any]]:
    """Retrieve all logged events that contain evidence video recordings.
    
    Returns:
        List[Dict[str, Any]]: List of event records with evidence video paths.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, session_id, timestamp, event_type, ear_value, mar_value, metadata FROM events "
        "WHERE metadata LIKE '%evidence%' OR event_type = 'evidence_recorded' ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

