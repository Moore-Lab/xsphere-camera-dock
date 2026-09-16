"""Put the dock root on sys.path so `camera_dock` imports when pytest runs.

pytest adds the *test* directory to sys.path, not the project root, so without
this `import camera_dock` fails when the suite is run from anywhere.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
