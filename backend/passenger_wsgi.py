"""
cPanel Passenger entrypoint.

In cPanel "Setup Python App":
- App startup file: passenger_wsgi.py
- Application entry point: application
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Ensure env file name for pydantic-settings ("env") can be found if present.
os.chdir(CURRENT_DIR)

from app.main import app as application  # noqa: E402

