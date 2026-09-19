# liveOD: moved from kexp into waxx

Status 2026-09-18: **done in code and tested offline; not yet run on the machine.**
liveOD is `waxx.util.live_od`. This file records what was done, the rules that still
apply, and what is left for a person.

## Why

waxx/waxa already owned the *protocol* — `waxx/base/expt.py` builds the INIT_RUN / END_RUN
payloads, `waxa/base/scribe.py` handles camera-ready and abort, `waxa/data/data_saver.py`
has the saver's "called by liveOD" half — but the implementation (15 files, ~5,300 lines)
was stranded in `kexp/util/live_od/`. A second lab using waxx would have needed a copy.

## How it is put together now

| Piece | Where | Notes |
|---|---|---|
| liveOD itself | `waxx/util/live_od/` | Same file names as before. Imports no lab package. |
| What a lab tells it | `waxx/util/live_od/config.py` — `LiveODConfig`, `set_config`, `get_config` | A registry, so no constructor signature changed. Every field is read somewhere; there are no placeholders. |
| The K machine's values | `k-exp/kexp/config/live_od.py` — `make_live_od_config()` | Data saver + run-id source, the five cameras and their bar order, default ROI ids, `ExptParams` as the params holder, the per-shot cross-section rule, the taskbar app id. Importing it has no side effects. |
| Launchers | `k-exp/kexp/util/live_od/gui/main_window.py`, `gui/remote_viewer_window.py` | `_bat/live_od.bat`, `_bat/live_od_viewer.bat`, both `.lnk`, and the documented `python -m kexp.util.live_od.gui.…` commands are unchanged. The window launcher builds the config and calls waxx's `main(config)`. The waxx window module refuses to run by itself and says why. |
| Old import paths | every other `kexp/util/live_od/**.py` | Aliases: the module object in `sys.modules` *is* the waxx module, so classes are identical, private names exist, patching one patches the other. `camera_mother` is a re-export rather than an alias because it keeps the legacy names `DATA_DIR` / `RUN_ID_PATH`. |

What changed inside the moved files (eight of fifteen are byte-identical apart from the
package prefix; `diff` against the last kexp commit shows the rest):

| File | Change |
|---|---|
| `camera_mother.py` | Params holder and camera lookup come from the config (were `kexp.config.expt_params` / `kexp.config.camera_id`). Dropped: an unused import of the Basler SDK; module-level `DATA_DIR`/`RUN_ID_PATH` (never used; `TypeError` on a PC with no data-directory variable); a `CameraNanny()` default argument built at import. `unpack_group` is imported only on the legacy path, from `waxa.atomdata_base`. |
| `camera_connection_widget.py` | The camera bar is built from `config.camera_params_list` (was five hard-coded K-machine cameras, and `from kexp import cameras`, which pulled in ARTIQ and the whole experiment stack). An unused `kexp.config.ip` import is gone (it touched the data drive and executed the device db at import). `ROISelector` stays; its latent bug (it called the `server_talk` *module*) is fixed. |
| `gui/main_window.py` | Takes the config; data saver, run-id source, grab-drain predicate, default ROI, app id and title come from it. Iterates `camera_conn_bar.buttons` instead of five attribute names. `main(config)` replaces the `__main__` block. The config check happens before any Qt object exists. |
| `live_od_server.py` | The Basler grab-drain gate asks `config.camera_needs_grab_drain` (default: key contains "basler"). New `shot_conditions_signal`. |
| `gui/analyzer.py`, `shot_cross_section.py` | Atom numbers are divided by a per-shot cross section from `config.cross_section_for_shot` and emitted at SHOT_COMPLETE. With no rule configured they stay integrated OD — waxx never assumes potassium. |
| `gui/fk_tof_window.py` | Redraws the fitted curve with the fit object's own model instead of re-deriving it with kamo's potassium mass (identical arithmetic). liveOD no longer imports kamo. |
| `live_od_client.py` | `shot_complete(..., shot_conditions=None)`; timeout message keeps the diagnosis it was given. |

Consumers repointed at waxx: `kexp/base/clients.py`, `kexp/util/profiling/startup_steps.py`,
and the agent tools `occupancy.py` (the load-by-file-path trick is gone: importing from waxx
is ~0.1 s and pulls in nothing), `reset_liveod.py`, `check_health.py`, `watch_run.py`.
Docs: `AGENTS.md` component table, `k-exp.wiki/Code-architecture`.

## Rules that still apply

1. **Never rename the beacon ids** `"live_od"` / `"live_od_broadcast"`. The experiment-side
   client, every remote viewer and the agent tools find liveOD by those literals and cannot
   see the config — which is why they are deliberately not config fields.
2. **The experiment process imports `waxx.util.live_od.live_od_client` and nothing else**
   from here. Both package `__init__`s stay free of module-level imports (tested).
3. **waxx must not import kexp** (tested by importing every module here in a fresh process).
4. **Never assign `self.live_od_client = None`** in an experiment: ARTIQ fails to compile
   with "cannot unify NoneType with LiveODClient" (`waxx/base/expt.py:40-42`).
5. **Never restart liveOD mid-run** — it holds the only copy of run state. Gate on `POLL`
   `run_in_progress == False`.
6. **Never start a real liveOD window or server in a test.** It beacons onto the lab
   network. Drive handlers through stand-in objects and the client through scripted
   replies; to build the real window, replace the thread `start()`s with no-ops
   (`k-exp/tests/test_live_od_migration.py` does all three).
7. The wire protocol carries only dicts, lists, primitives and numpy — no instance of a
   custom class. Keep it that way: pickle would embed the class's module path, and old and
   new clients/servers could no longer be mixed.

## Tests (all offline)

- `k-exp/tests/test_live_od_migration.py` — every old module path exists and gives the
  same objects (the full pre-move public surface); aliases are the waxx module itself;
  load-by-file-path still works; waxx imports no kexp; the client import is light; the
  window refuses to start without a config and the waxx module is not runnable alone; the
  camera bar follows the config (offscreen Qt); `DataHandler` takes its params holder and
  camera table from the config; **the real `LiveODWindow` builds with kexp's real config**
  with thread starts disabled, and a shot's atom number comes out tagged `high-field`.
- `k-exp/tests/test_live_od_reset.py`, `test_live_od_shot_cross_section.py`,
  `test_live_od_config_builder.py`; `wax/waxx-src/tests/test_live_od_config.py`.

What they cannot show is the real process, cameras and data drive.

## Left for a person

- [ ] **Restart liveOD** (not during a run) and go through the smoke set below.
- [ ] `.claude/hooks/lab_guard.py:82` guards edits to `kexp/util/live_od/live_od_server.py`
      by path. That file is now a 13-line alias; the real one is
      `waxx/util/live_od/live_od_server.py` and is **not covered**. The hook was left alone
      on purpose (a safety hook is the owner's to change): add the waxx path to that regex.
- [ ] This directory still holds two stray 1 GB `dataset_db.mdb` (+ `.mdb-lock`, one under
      `gui/`) from an `artiq_master` once run here, and `__pycache__` from an abandoned
      earlier port (`camera_server`, `protocol`, `viewer_client`, …). Git ignores them
      (`*.mdb`), they do no harm, and they were not deleted because deleting is not
      reversible. Remove them when convenient.
- [ ] Git history: the moved files start fresh in wax (`git mv` cannot cross repos). Say so
      in the commit message, or replay history with `git filter-repo`.
- [ ] Remaining prose mentions of the old location: `k-exp.wiki` (LiveOD page,
      Networking-intro worked example), `.claude/skills/run-experiment/SKILL.md`,
      `.claude/lab-rules.md`. The launch commands they give are still correct.

### Smoke set
- Basler absorption run with `save_data=True`: run id advances, HDF5 complete, images in order.
- Andor run: the camera opens on demand.
- `setup_camera=False` run; `suppress_live_od=True` tool run.
- Reset from the local button, a remote viewer and `reset_liveod.py` — during the camera
  wait, mid-scan, between runs.
- A remote viewer on another PC, **not restarted**: images, scalars, adjust panel, camera
  buttons. (Its camera buttons now list z before x: there is one camera order, the local
  bar's.)
- One high-field and one low-field run: `atom_cross_section_source` in the scalars reads
  `high-field` / `low-field-uncalibrated`, and liveOD's atom number matches atomdata's.
- FK-TOF window on a multi-image run: temperature and curve as before.
- `art mot_tof.py`: the three liveOD rows still appear; `occupancy.py` reports free.

## Later, if wanted
- `waxx.base.Expt` could construct the client itself (it needs only beacon + `hardware_id`;
  no import cycle). The `suppress_live_od` / `setup_camera=False` policy in
  `kexp/base/clients.py` would move with it, and rule 4 applies.
- A camera-type registry in `camera_nanny` instead of the `"basler"` / `"andor"` chains.
- `beacon/basler/` is a parallel, independent Basler server/GUI — a consolidation target.
- `pickle.loads` on an unauthenticated LAN socket is arbitrary code execution for anything
  on the subnet. Pre-existing; worth fixing before another lab depends on this.
- The per-shot cross-section rule kexp passes lives in `waxa.calibrations.cross_section`
  and switches on the K machine's coil current: K-machine policy in a generic package.
