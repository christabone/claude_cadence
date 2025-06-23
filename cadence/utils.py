"""
General utility functions for Claude Cadence

This module contains shared utility functions used across the system.
"""

import uuid
from datetime import datetime


def generate_session_id() -> str:
    """
    Generate a unique session ID with timestamp and UUID.

    Format: YYYYMMDD_HHMMSS_[8-char-uuid]
    Example: 20240103_143022_a1b2c3d4

    Returns:
        str: A unique session identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"
