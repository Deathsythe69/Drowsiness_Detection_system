"""Database Module.

Manages connection lifecycle and SQLite schema setup.
"""

import sqlite3
import os

DB_PATH = "drowsiness_tracker.db"

def get_db_connection() -> sqlite3.Connection:
    """Open and return a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Database connection object.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_label TEXT,
        start_time TEXT,
        end_time TEXT,
        baseline_ear REAL
    )
    """)
    
    # 2. Create events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        timestamp TEXT,
        event_type TEXT,
        ear_value REAL,
        mar_value REAL,
        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()
