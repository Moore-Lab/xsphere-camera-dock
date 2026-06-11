# xsphere-camera-dock — Development Session Log

Running, chronological record of development for **this repo only**
(`xsphere-camera-dock`). Changes inside a camera submodule are logged in *that
submodule's* own `docs/session-log.md` — write to the log of the repo you're modifying.

Scope of this log: the shared camera interface (`camera_dock/`), the dock integration
layer (combined feeds, unified controls, snapshot/record across cameras), camera
submodule pointer bumps, and (eventually) the dock's web UI.

Newest entries first. Keep entries short and factual; convert relative dates to absolute.
See the [README](../README.md) for this repo's roadmap.

---

## 2026-06-11 — Session log started; baseline

**Context.** First logged session. Established per-repo session logs across the project.

**State of this repo:**

- `camera_dock/base.py` — `CameraBase`, a `runtime_checkable` structural `Protocol`
  (`connect` / `disconnect` / `set_exposure` / `set_frame_rate` / `grab` / `frames`).
  Dependency arrow points dock → cameras (cameras don't import the dock), avoiding a
  circular submodule dependency. **Defined; no integration layer yet.**
- Submodules: `basler-acA1440` (driver + test GUI implemented — see its log),
  `zelux-cs165mu` (scaffolding only).

**Next (dock):**

1. Once both camera drivers are validated in isolation, build the integration layer over
   `CameraBase`: combined live feeds, unified frame-rate / exposure controls, synchronized
   snapshot and recording.
2. Surface the dock as a web app feeding up into the top-level `xsphere-daq` panel.
