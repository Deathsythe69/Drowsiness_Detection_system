"""Unit tests for persistent user profiles and calibration storage."""

import os
from storage.db import init_db
from storage.models import UserProfile

def test_user_profile_crud():
    """Verify SQLite creation, loading, update, and deletion of UserProfile."""
    init_db()
    
    # 1. Save new profile
    profile = UserProfile.save_or_update(
        name="TestUser_Glasses",
        baseline_ear=0.310,
        ear_threshold=0.217,
        baseline_mar=0.220,
        mar_threshold=0.620,
        blink_rate=14.5
    )
    assert profile.id is not None
    assert profile.name == "TestUser_Glasses"
    assert profile.ear_threshold == 0.217
    
    # 2. Retrieve all profiles
    profiles = UserProfile.get_all()
    assert any(p.name == "TestUser_Glasses" for p in profiles)
    
    # 3. Update existing profile
    updated = UserProfile.save_or_update(
        name="TestUser_Glasses",
        baseline_ear=0.320,
        ear_threshold=0.224
    )
    assert updated.id == profile.id
    assert updated.ear_threshold == 0.224
    
    # 4. Delete profile
    UserProfile.delete(profile.id)
    profiles_after = UserProfile.get_all()
    assert not any(p.name == "TestUser_Glasses" for p in profiles_after)
