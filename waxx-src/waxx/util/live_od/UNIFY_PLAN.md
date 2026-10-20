# liveOD: one program, server on the host, viewer everywhere else

Status 2026-10-20: **plan only; nothing in this file is built.** Phase 1 (the viewer brought
level with the acquisition window, and the person hold shown in both) is done in code
and tested offline (`tests/test_liveod_viewer_parity.py`). It has not been run against
the live server yet.

## Goal

A person starts "liveOD" the same way on any lab PC. On the host (kong) it is the
acquisition server: cameras, run file writer, REP/PUB ports, the analyzer. Everywhere
else it is a viewer of that server. Both roles show the same window. The only things
that differ are the ones that need the hardware, and each says so where it would be
used.

## Where things stand (after phase 1)

| | Acquisition window (`gui/main_window.py`) | Viewer (`gui/remote_viewer_window.py`) |
|---|---|---|
| Status strip: pill, run label, progress, Δt/ETA, next run id, stall watchdog | in-process signals | same `StatusStrip`, fed by RUN_STARTED (+`info`), RUN_STATE, SHOT_PROGRESS, SHOT_TIMING, NEXT_RUN_ID |
| Late join | n/a | one POLL + GET_LOG on every (re)connection |
| Abort | `reset()` when a run is active | POLL, then RESET `{source: person, run_id}`; nothing sent with no run |
| Person hold | `HoldBadge` (polls the monitor server; one click releases, no confirm) | same |
| Adjust | `adjust_specs_signal` | ADJUST_SPECS broadcast (+ GET_ADJUST_VALUES at connect) |
| FK TOF params, pixel size, xvar ranges | in-process | RUN_STARTED `info` |
| Settings (geometry, toggles, theme, OD levels) | `QSettings("waxx","live_od")` | `QSettings("waxx","live_od_viewer")` |
| Camera ⚙ settings, 🎥 live view, Persist | in-process `CameraHost` | **missing** |
| ROI that the atom number uses | this window's ROI (or its zoom) | **its own ROI is display-only**; the numbers it shows are the server's |
| Scalar subscriptions | in-process counter | SUBSCRIBE/UNSUBSCRIBE with no lease (a crashed viewer leaks a count) |

## What it would take

### 1. Split ownership from display (the big refactor)

`LiveODWindow.__init__` builds everything itself: the camera nanny/host, the hidden
`CamConnBar` that owns the cameras, the `Analyzer`, `LiveODServer`, `LiveODBroadcaster`,
and the run-thread plumbing (`spawn_baby`, `on_run_done`, `on_run_exited`, `reset`,
`shutdown`). Move all of it into a non-widget **`LiveODHost`** (a QObject). The window
then becomes a shell that holds a host only in the server role.

- `LiveODHost` owns: server, broadcaster, camera host + bridge, `CamConnBar`,
  `CameraNanny`/`CameraMother`, `CameraBaby`/`DataHandler`, Analyzer, the `server_talk`
  run-id poll, the shutdown hooks and `console_guard`.
- Keep the names that tests patch in `gui/main_window.py`'s namespace (`LiveODServer`,
  `LiveODBroadcaster`, `LiveODPlotter`, `CameraBaby`, `DataHandler`, `QMessageBox`,
  `QSettings`, `atexit`, `console_guard`) by importing them there and passing them in,
  or by moving the tests' patch targets in the same change.
- The closing-during-a-run confirm stays with the window; the host's `shutdown` does
  not change.

### 2. One feed interface, two implementations

The shell talks to a **feed**: signals in, actions out.

| Signal (in) | Local feed | Remote feed |
|---|---|---|
| run_started(info) | `spawn_baby` | RUN_STARTED (+`info`) |
| run_state, shot_progress, shot_timing, run_done | server signals | RUN_STATE, SHOT_PROGRESS, SHOT_TIMING, RUN_DONE |
| od_image, shot_scalars, fk_tof | Analyzer | OD_IMAGE, SHOT_SCALARS, FK_TOF |
| log | `QtLogHandler` | LOG_MSG (+ GET_LOG at connect) |
| cameras, markers, adjust specs/values, next run id | in-process | CAMERA_STATE, MARKERS, ADJUST_*, NEXT_RUN_ID |

| Action (out) | Local | Remote |
|---|---|---|
| abort | `reset(source="person")` | RESET with `run_id` |
| camera toggle | `CamConnBar` / host decision | CAMERA_CONTROL |
| markers, adjust value/spec, scalar subscribe | direct calls | SET_MARKERS, SET_ADJUST_*, SUBSCRIBE_SCALARS |
| camera settings ⚙, live view 🎥 | `CameraHost` in-process | beacon `camera_server:<host>:liveod`: `describe`/`snapshot`/live settings, `V2StreamSource` for 🎥 (already allowed from other PCs between runs; refused under a run's lock) |
| Persist, snap, rename, relinquish, EM-gain unlock | in-process | **not offered** (the camera host refuses them from other PCs, by design) |

`CameraSettingsDialog` and `LiveViewWindow` already take a duck-typed host (`gui/demo.py`'s
`DemoHost` proves it), so the remote ⚙/🎥 is a `RemoteCameraHost` adapter over the
beacon client. That adapter is the one new piece of camera code.

### 3. Pick the role at start

One launcher (`_bat/live_od.bat`, `python -m kexp.util.live_od`) calls
`waxx.util.live_od.app.main(config)`:

1. `--server` / `--viewer` on the command line decide it outright.
2. Otherwise: **server** only if this PC is the configured host (`LiveODConfig.server_hosts`,
   e.g. `("kong",)` by hostname) **and** no `live_od:<hw>` beacon is already answering.
   Anything else is a viewer.
3. A second start on kong while the server runs therefore opens a viewer instead of
   fighting over camera locks. liveOD has no duplicate-server check today. Add one in
   the server role as well: refuse to start when `live_od:<hw>` answers a POLL, the
   way the monitor server refuses a duplicate.

Hostname rather than "cameras present" or "data drive mapped": it is deterministic,
and a viewer PC with the data drive mapped must not become a second server.
`live_od_viewer.bat` and the `remote_viewer_window` module stay as shims (`--viewer`).

### 4. Decisions that are a person's

- **The ROI the live atom number uses.** Today it is the acquisition window's drawn ROI,
  or, with none, its **current zoom** (`analyzer.py:288-317`). In one program with many
  windows that cannot stay a property of one window. Either (a) make the ROI server
  state, set from any window and broadcast like markers, with no zoom fallback (no ROI
  means the full frame), or (b) keep it the host window's and label every viewer's
  numbers "ROI: the host's". (a) changes the live numbers people are used to.
  These numbers are display only: nothing in the run file comes from them, and
  `atomdata` recomputes from the raw frames with its own ROI.
- **Who may Abort from a viewer.** Anyone can now. Abort by run id cannot hit the
  next run, but there is still no authentication on either port, and both ports
  `pickle.loads` whatever arrives. Moving to JSON or msgpack plus a host allow-list
  is its own job and is not required for the merge.
- **Viewer on kong itself** (a second window beside the server): allowed by step 3.
  Is it wanted?

### 5. Smaller protocol gaps still open

- CAMERA_STATE carries legacy words and `persist` only. Add the host snapshot fields
  (`host_state`, `holder`, `n_subs`) so a remote `CameraControl(expect_snapshots=True)`
  greys and badges like the local one.
- Scalar subscriptions: give them a lease (re-sent every few seconds, expired after
  ~10 s) so a crashed viewer does not keep the server computing fits.
- AUX_NOTICE (diagnostic frames) is broadcast but shown by neither window. kexp's
  diagnostic viewer consumes it. Leave as is.
- ADJUST_SPECS is broadcast at run start only; a SET_ADJUST_SPEC from one viewer
  reaches the others at the next run. Re-broadcast on change.

## Phases

1. **Done 2026-10-20 (offline):** viewer parity listed above; `HoldBadge` in both
   windows; broadcaster RUN_STARTED `info`, SHOT_TIMING, NEXT_RUN_ID, ADJUST_SPECS;
   POLL `save_data`. Owed: a check against the live server on kong (start a viewer on
   another PC mid-run; Abort from it; hold on/off from the queue panel).
2. Extract `LiveODHost` from `LiveODWindow` (no behaviour change; the window tests
   must pass unchanged apart from patch targets).
3. Feed interface; the viewer becomes the shell + `RemoteFeed`; delete the duplicated
   viewer code.
4. One launcher with role selection and the duplicate-server refusal.
5. `RemoteCameraHost` adapter: ⚙ and 🎥 from a viewer.
6. Section 4's ROI decision, then section 5's gaps.

Phases 2 and 3 are the risky ones: the acquisition window is the data path. Each is
one branch, with the full `tests/` suite and a supervised run on kong before merging.
