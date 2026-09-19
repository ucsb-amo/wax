"""What a lab has to tell liveOD.

liveOD lives in waxx and knows nothing about any lab. Everything lab-specific --
where data goes, which cameras exist, which cross section applies -- comes from
one LiveODConfig. The lab's launcher builds it and calls ``set_config`` before
creating any liveOD object (kexp: kexp/config/live_od.py, called from
kexp/util/live_od/gui/main_window.py); the liveOD classes read it with
``get_config``. A registry rather than constructor arguments, so every existing
constructor signature keeps working.

Every field below is read somewhere in this package; there are no placeholders.
Two things are deliberately NOT configurable: the beacon ids "live_od" and
"live_od_broadcast". The experiment-side client, every remote viewer and the agent
tools find liveOD by those literals and cannot see this config, so a field for
them could only break discovery.

Stdlib-only on purpose: the experiment process may import this package, and waxx
must not import a lab package (see waxx/util/comms_server/hardware_id.py).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ImageCounts:
    """The three numbers DataHandler keeps about a run. Default for
    ``LiveODConfig.params_factory``; a lab may hand in its own ExptParams class
    instead (kexp does), as long as it has these attributes. Other run
    parameters are set on the instance as they arrive."""
    N_img: int = 1
    N_shots: int = 1
    N_pwa_per_shot: int = 1


def _basler_needs_grab_drain(camera_key: str) -> bool:
    return "basler" in camera_key


@dataclass
class LiveODConfig:
    # ---- data plumbing (the acquisition window needs both) ------------------
    # waxa DataSaver: reserves the run id and file at INIT_RUN, saves at END_RUN.
    data_saver: Any = None
    # waxa server_talk: get_run_id(), update_run_id(), check_for_mapped_data_dir()
    run_id_source: Any = None

    # ---- cameras -------------------------------------------------------------
    # One button per entry, in this order, in the window's camera bar; also the
    # order of the CAMERA_STATE broadcast that remote viewers build buttons from.
    camera_params_list: List[Any] = field(default_factory=list)
    # Keys of cameras to open as soon as the window starts. Default: none.
    cameras_open_on_start: List[str] = field(default_factory=list)
    # camera key -> CameraParams, or None. DataHandler's fallback when a run's
    # INIT_RUN payload carries no camera_params.
    resolve_camera_params: Callable[[str], Optional[Any]] = lambda key: None
    # Cameras whose previous grab loop must fully exit before the next run may
    # arm the hardware (the WAIT_CAM_READY gate and the window's wiring for it).
    camera_needs_grab_drain: Callable[[str], bool] = _basler_needs_grab_drain
    # camera key -> saved-ROI id to start from, or None for no default.
    default_roi_id_for: Callable[[str], Optional[str]] = lambda key: None

    # ---- run bookkeeping -----------------------------------------------------
    params_factory: Callable[[], Any] = ImageCounts

    # ---- physics (species) ---------------------------------------------------
    # shot_conditions -> (cross section in m^2, source tag). shot_conditions is
    # the {key: float} dict the experiment sends with SHOT_COMPLETE (every
    # single-valued DataVault container; empty or None from an older experiment
    # process). Integrated OD x area is divided by the result to show an atom
    # number. kexp passes the analysis's own rule, which switches on the recorded
    # outer-coil current. None: no atom number, the scalar stays integrated OD.
    cross_section_for_shot: Optional[Callable[[Optional[dict]], Any]] = None

    # ---- identity of the acquisition window ------------------------------------
    # Windows taskbar identity. A new value un-groups existing pinned buttons.
    app_user_model_id: str = "waxx.live_od"
    window_title: str = "LiveOD Server"


# ---------------------------------------------------------------------------
# The active config
# ---------------------------------------------------------------------------

_active = None


def set_config(config):
    """Make ``config`` the one liveOD objects in this process read. Call it once,
    in the launcher, before creating the window."""
    global _active
    _active = config
    return config


def get_config():
    """The active config. With none set, a default one: no cameras known, no
    atom-number calibration, no data saver -- enough for the remote viewer and
    for tests, not for the acquisition window (which checks)."""
    global _active
    if _active is None:
        _active = LiveODConfig()
    return _active
