"""Unit tests for AsyncDBLogger non-blocking database queue worker."""

import time
import pytest
from storage.db import init_db
from storage.models import Session, Event
from ui.main_window import AsyncDBLogger


def test_async_db_logger_non_blocking():
    """Verify AsyncDBLogger commits events asynchronously without blocking."""
    init_db()
    session = Session.create(user_label="TestUser", baseline_ear=0.25)
    assert session.id is not None

    logger = AsyncDBLogger()
    
    # Push 5 events in quick succession
    for i in range(5):
        logger.log_event(session.id, f"test_event_{i}", 0.22, 0.15, f"Meta {i}")

    # Allow worker thread brief window to flush
    time.sleep(0.3)
    logger.stop()

    events = Event.get_by_session(session.id)
    event_types = [e.event_type for e in events]
    for i in range(5):
        assert f"test_event_{i}" in event_types
