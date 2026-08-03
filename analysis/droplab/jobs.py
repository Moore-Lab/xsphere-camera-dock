"""Background analysis jobs for droplab.

Detection / tracking / distribution runs take seconds-to-minutes, so HTTP
handlers only enqueue them here. A small thread pool (2 workers — the stages
are CPU/decode bound) executes jobs; the browser polls GET /jobs/{id}.

Cancellation is cooperative: cancel() sets an Event that long loops
(run_detection, sum-view, distance scan) check between frames.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional


class Job:
    def __init__(self, kind: str, video: str, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.video = video
        self.params = params
        self.state = "queued"            # queued | running | done | error | cancelled
        self.progress = 0.0              # 0..1
        self.message = ""
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self._cancel = threading.Event()

    # -- used by workers ----------------------------------------------------
    def report(self, done: int, total: int, message: str = "") -> None:
        self.progress = min(done / total, 1.0) if total else 0.0
        if message:
            self.message = message

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- API shape ----------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "video": self.video,
            "state": self.state, "progress": round(self.progress, 4),
            "message": self.message, "result": self.result, "error": self.error,
            "params": self.params,
            "elapsed_s": round((self.finished or time.time()) - (self.started or time.time()), 2)
                         if self.started else 0.0,
        }


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="droplab-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit_exclusive(self, kind: str, video: str, params: dict,
                         fn: Callable[[Job], dict]):
        """Register + enqueue a job unless the video already has an active one.

        The check and the registration happen under one lock, so two racing
        requests can never both start jobs on the same video. Returns
        (job, None) on success or (None, active_job) when rejected.
        """
        job = Job(kind, video, params)
        with self._lock:
            for other in self._jobs.values():
                if other.video == video and other.state in ("queued", "running"):
                    return None, other
            self._jobs[job.id] = job

        def run():
            if job.cancelled():
                if job.state == "queued":       # cancel() may have marked it
                    job.state = "cancelled"
                    job.finished = time.time()
                return
            job.state = "running"
            job.started = time.time()
            try:
                job.result = fn(job)
                job.state = "cancelled" if job.cancelled() else "done"
                job.progress = 1.0 if job.state == "done" else job.progress
            except Exception as e:           # surfaced to the browser, not lost
                job.state = "error"
                job.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job, None

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is not None and job.state in ("queued", "running"):
            job._cancel.set()
            with self._lock:
                if job.state == "queued":   # free the video's idle slot now,
                    job.state = "cancelled"  # not when the pool gets around to it
                    job.finished = time.time()
        return job

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel what can be cancelled and wait briefly for the pool to drain.

        Called on server shutdown (also by the DAQ panel, via
        ``app.state.shutdown_hooks``). Worker threads are non-daemon and the
        stages write result files directly, so we never kill one mid-write —
        if a stage outlives the timeout we say so instead of leaving the
        launcher window silently blocked on interpreter exit.
        """
        with self._lock:
            for job in self._jobs.values():
                if job.state in ("queued", "running"):
                    job._cancel.set()
                    if job.state == "queued":
                        job.state = "cancelled"
                        job.finished = time.time()
        drain = threading.Thread(
            target=lambda: self._pool.shutdown(wait=True, cancel_futures=True),
            name="droplab-drain", daemon=True)
        drain.start()
        drain.join(timeout)
        if drain.is_alive():
            print("[droplab] an analysis stage is still finishing — "
                  "leaving it to write its output rather than truncating it…",
                  flush=True)

    def list(self, video: Optional[str] = None) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.as_dict() for j in jobs if video is None or j.video == video]

    def active_for(self, video: str) -> Optional[Job]:
        with self._lock:
            for j in self._jobs.values():
                if j.video == video and j.state in ("queued", "running"):
                    return j
        return None
