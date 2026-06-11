# xsphere-camera-dock

Centralized location for connecting to and interfacing with the experiment's cameras.

The dock combines individual, per-camera modules (each a git submodule) into a single
control surface: synchronized configuration, combined live feeds, snapshots, and
recording. It is itself a submodule of [`xsphere-daq`](https://github.com/Moore-Lab/xsphere-daq),
the broader data-acquisition / experiment-control project.

## Submodules

| Path | Camera | Repo |
|------|--------|------|
| `basler-acA1440` | Basler acA1440-220um | [Moore-Lab/basler-acA1440](https://github.com/Moore-Lab/basler-acA1440) |
| `zelux-cs165mu`  | Thorlabs Zelux CS165MU | [Moore-Lab/zelux-cs165mu](https://github.com/Moore-Lab/zelux-cs165mu) |

Each camera module exposes the same conceptual driver interface
(`connect` / `set_exposure` / `set_frame_rate` / `grab` / `frames`), so the dock can
drive heterogeneous cameras uniformly.

```bash
git clone --recurse-submodules https://github.com/Moore-Lab/xsphere-camera-dock.git
# or, after a plain clone:
git submodule update --init --recursive
```

## Roadmap

1. Get each camera module working well in isolation (driver + test GUI).
2. Define a shared camera interface (the `CameraBase` protocol the dock expects).
3. Combine feeds in the dock: unified controls for **frame rate**, **exposure**,
   **snapshots**, and **recording** across all cameras.
4. Surface it as a web app so feeds stream into an embedded window and the rig can be
   driven remotely — feeding up into the top-level `xsphere-daq` control panel.

## Status

Scaffolding only — submodule structure in place, integration layer pending. Development
is tracked in [`docs/session-log.md`](docs/session-log.md).
