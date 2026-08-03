#!/usr/bin/env python3
"""
play_video.py — a simple video player for recordings (.avi / .mp4).
===================================================================
A no-frills player for the videos in the repo's recordings/ folder: pick a file,
watch it, play/pause, stop (back to the start), and skip forward/back — by a
single frame or by a few seconds. Built on PyQt5 (the same toolkit the camera
GUIs use) plus OpenCV for decoding.

For frame-by-frame analysis (intensity, motion, timing, zoom) use
``analyze_frames.py --view`` instead — this script is just for watching.

Run it:
    python analysis/play_video.py                 # opens a chooser in recordings/
    python analysis/play_video.py recordings/foo.avi   # opens straight into a file
"""

import argparse
import os
import sys

import cv2
import numpy as np

# recordings/ resolved from THIS file (repo root is the parent of analysis/), so
# the default works no matter what directory the player is launched from.
DEFAULT_RECORDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "recordings"))

SKIP_SECONDS = 5      # how far the ⏪ / ⏩ buttons jump


def launch(initial_path=None, start_dir=DEFAULT_RECORDINGS_DIR):
    from PyQt5 import QtCore, QtGui, QtWidgets

    class Player(QtWidgets.QMainWindow):
        def __init__(self, path=None):
            super().__init__()
            self.setWindowTitle("Video Player — recordings")
            self.resize(960, 720)

            self.cap = None
            self.total = 0
            self.fps = 30.0
            self.cur_idx = -1
            self.cur_frame = None

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._on_tick)

            self._build_ui()
            if path:
                self._load(path)

        # ── UI ──────────────────────────────────────────────────────────
        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            top = QtWidgets.QHBoxLayout()
            open_btn = QtWidgets.QPushButton("Open video…")
            open_btn.clicked.connect(self._choose_file)
            top.addWidget(open_btn)
            self.file_label = QtWidgets.QLabel("No file loaded")
            self.file_label.setStyleSheet("color:#888;")
            top.addWidget(self.file_label, 1)
            root.addLayout(top)

            self.video_label = QtWidgets.QLabel("Open a video to begin")
            self.video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.video_label.setStyleSheet("background:#151515; color:#aaa;")
            self.video_label.setMinimumSize(1, 1)
            self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
            root.addWidget(self.video_label, 1)

            # Scrubber + time readout
            row = QtWidgets.QHBoxLayout()
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setEnabled(False)
            self.slider.sliderMoved.connect(self.goto)
            self.slider.valueChanged.connect(self._on_slider)
            row.addWidget(self.slider, 1)
            self.time_label = QtWidgets.QLabel("--:-- / --:--")
            self.time_label.setStyleSheet("font-family:monospace;")
            row.addWidget(self.time_label)
            root.addLayout(row)

            # Transport buttons
            btns = QtWidgets.QHBoxLayout()
            self.b_stop = QtWidgets.QPushButton("⏹ Stop")
            self.b_back_s = QtWidgets.QPushButton(f"⏪ −{SKIP_SECONDS}s")
            self.b_back_f = QtWidgets.QPushButton("◀ −1")
            self.b_play = QtWidgets.QPushButton("▶ Play")
            self.b_fwd_f = QtWidgets.QPushButton("+1 ▶")
            self.b_fwd_s = QtWidgets.QPushButton(f"+{SKIP_SECONDS}s ⏩")
            self.b_stop.clicked.connect(self.stop)
            self.b_back_s.clicked.connect(lambda: self.skip_seconds(-SKIP_SECONDS))
            self.b_back_f.clicked.connect(lambda: self.step(-1))
            self.b_play.clicked.connect(self.toggle_play)
            self.b_fwd_f.clicked.connect(lambda: self.step(1))
            self.b_fwd_s.clicked.connect(lambda: self.skip_seconds(SKIP_SECONDS))
            self._buttons = [self.b_stop, self.b_back_s, self.b_back_f,
                             self.b_play, self.b_fwd_f, self.b_fwd_s]
            for b in self._buttons:
                b.setEnabled(False)
                btns.addWidget(b)
            root.addLayout(btns)

            self.statusBar().showMessage(
                "Space: play/pause   ← →: ±1 frame   "
                f"Shift+← →: ±{SKIP_SECONDS}s   Home: stop")

            def sc(seq, fn):
                QtWidgets.QShortcut(QtGui.QKeySequence(seq), self).activated.connect(fn)
            sc(QtCore.Qt.Key_Space, self.toggle_play)
            sc(QtCore.Qt.Key_Left, lambda: self.step(-1))
            sc(QtCore.Qt.Key_Right, lambda: self.step(1))
            sc("Shift+Left", lambda: self.skip_seconds(-SKIP_SECONDS))
            sc("Shift+Right", lambda: self.skip_seconds(SKIP_SECONDS))
            sc(QtCore.Qt.Key_Home, self.stop)

        # ── File loading ────────────────────────────────────────────────
        def _choose_file(self):
            start = start_dir if os.path.isdir(start_dir) else os.getcwd()
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open video", start,
                "Videos (*.avi *.mp4);;All files (*)")
            if path:
                self._load(path)

        def _load(self, path):
            self.pause()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                QtWidgets.QMessageBox.critical(
                    self, "Open failed", f"OpenCV could not open:\n{path}")
                return
            if self.cap is not None:
                self.cap.release()
            self.cap = cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else 30.0
            self.total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            self.file_label.setText(
                f"{os.path.basename(path)}   —   {self.total} frames @ {self.fps:.1f} fps")
            for b in self._buttons:
                b.setEnabled(True)
            self.slider.setEnabled(True)
            self.slider.blockSignals(True)
            self.slider.setRange(0, self.total - 1)
            self.slider.blockSignals(False)
            self.cur_idx = -1
            self.goto(0)

        # ── Frame access / display ──────────────────────────────────────
        def _read_at(self, idx):
            if self.cap is None:
                return None
            idx = max(0, min(idx, self.total - 1))
            if idx != self.cur_idx + 1:            # avoid a re-seek on sequential play
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            return frame if ok else None

        def goto(self, idx):
            idx = max(0, min(int(idx), self.total - 1))
            frame = self._read_at(idx)
            if frame is None:
                return
            self.cur_frame = frame
            self.cur_idx = idx
            self._render()
            self._sync_controls()

        def step(self, delta):
            self.goto(self.cur_idx + delta)

        def skip_seconds(self, secs):
            self.goto(self.cur_idx + int(round(secs * self.fps)))

        def _on_slider(self, val):
            if val != self.cur_idx:
                self.goto(val)

        def _sync_controls(self):
            self.slider.blockSignals(True)
            self.slider.setValue(self.cur_idx)
            self.slider.blockSignals(False)
            self.time_label.setText(
                f"{self._fmt_t(self.cur_idx)} / {self._fmt_t(self.total - 1)}"
                f"   [{self.cur_idx}/{self.total - 1}]")

        def _fmt_t(self, idx):
            s = idx / self.fps if self.fps else 0
            return f"{int(s // 60):02d}:{s % 60:04.1f}"

        def _render(self):
            if self.cur_frame is None:
                return
            frame = self.cur_frame
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix.scaled(
                self.video_label.width(), self.video_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._render()

        # ── Playback ────────────────────────────────────────────────────
        def toggle_play(self):
            if self.cap is None:
                return
            self.pause() if self.timer.isActive() else self.play()

        def play(self):
            if self.cur_idx >= self.total - 1:      # restart if parked at the end
                self.goto(0)
            self.timer.start(max(1, int(1000.0 / self.fps)))
            self.b_play.setText("❚❚ Pause")

        def pause(self):
            self.timer.stop()
            self.b_play.setText("▶ Play")

        def stop(self):
            self.pause()
            self.goto(0)

        def _on_tick(self):
            if self.cur_idx >= self.total - 1:
                self.pause()
                return
            self.step(1)

        def closeEvent(self, event):
            self.pause()
            if self.cap is not None:
                self.cap.release()
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Player(initial_path)
    win.show()
    return app.exec_()


def main(argv=None):
    p = argparse.ArgumentParser(description="Simple player for recordings (.avi/.mp4).")
    p.add_argument("path", nargs="?", help="a video file to open (optional)")
    args = p.parse_args(argv)
    initial = args.path if args.path and os.path.isfile(args.path) else None
    start_dir = (os.path.dirname(args.path) if args.path and os.path.dirname(args.path)
                 else DEFAULT_RECORDINGS_DIR)
    return launch(initial_path=initial, start_dir=start_dir)


if __name__ == "__main__":
    sys.exit(main())
