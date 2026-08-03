"""droplab — browser-based droplet analysis for camera-dock recordings.

A FastAPI server (port 8020 by default) around the Moore-Lab count-droplets
pipeline: detection preview/tuning with drag-ROI, background detection runs,
track linking, distribution fits, and longitudinal chunk analysis — all
operating on the dock's recordings/ folder so videos can be analyzed while
the next one records in another browser window.

Run:  python -m analysis.droplab.server
"""
