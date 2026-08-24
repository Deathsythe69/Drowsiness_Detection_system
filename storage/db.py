"""Database Module.

Manages connection lifecycle, schema migrations, and SQLite setup
including session telemetry, events, and persistent user profiles.
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
    """Create database tables and apply schema updates if they do not exist."""
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
    
    # 2. Create events table with optional metadata column
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        timestamp TEXT,
        event_type TEXT,
        ear_value REAL,
        mar_value REAL,
        metadata TEXT DEFAULT '',
        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
    )
    """)
    
    # 3. Create user_profiles table for persistent personalized calibration
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        baseline_ear REAL NOT NULL,
        ear_threshold REAL NOT NULL,
        baseline_mar REAL NOT NULL,
        mar_threshold REAL NOT NULL,
        blink_rate REAL DEFAULT 15.0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    
    # 4. Schema Migration: Ensure 'metadata' column exists in events table
    cursor.execute("PRAGMA table_info(events)")
    columns = [row[1] for row in cursor.fetchall()]
    if "metadata" not in columns:
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN metadata TEXT DEFAULT ''")
        except Exception as e:
            print(f"Migration error adding metadata column: {e}")
    
    conn.commit()
    conn.close()
