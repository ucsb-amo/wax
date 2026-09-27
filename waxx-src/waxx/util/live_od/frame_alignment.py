"""Do a run's frames sit in their shots' slots? A check from arrival times alone.

liveOD files frame ``i`` of a run as image ``i % per_shot`` of shot ``i // per_shot``
(``per_shot = N_img / N_shots``). Nothing on a frame says which trigger made it, so
a stray edge (or a frame left over from before the run) followed later by a missed
trigger leaves the count right and every frame in between one slot off. Counting
cannot see that; arrival times sometimes can.

What the times can prove, and why
---------------------------------
The server records each SHOT_COMPLETE (``shot_t``) when the message arrives. The
experiment sends it from ``cleanup_scan_kernel`` *after* ``put_shot_data`` has
waited for the RTIO timeline (``core.wait_until_mu(now_mu())``), as a blocking RPC:
the kernel does not schedule a single event of the next shot until the server has
replied. So every frame of shot ``s`` is caused by a trigger that fired after the
server recorded ``shot_t[s-1]``, and it reaches the host after that trigger's
exposure and readout. For shot 0 the same holds with ``t_start``, the moment the
server told the experiment the camera was ready (``init_kernel`` waits for that
before the scan). A frame in the slot of shot ``s`` that arrived *before* that
bound cannot be shot ``s``'s: an extra frame came earlier and the slots from there
on hold the wrong shots. That is the only thing this module calls an issue.

What they cannot prove
----------------------
* A frame that arrives *late* (after its shot was reported complete) may be a
  frame of the next shot (a trigger was missed earlier), or just a slow one: the
  camera thread and the image dispatcher share the GIL with h5py writes to a
  network drive, which can stall them for seconds. Late frames are reported as
  notes (a WARNING), never as issues.
* A one-frame shift can hide at every shot boundary: the last frame of a shot is
  often still being read out when SHOT_COMPLETE arrives, so the frame that moved
  into the next shot's slot arrives right at the boundary, where both readings
  are possible. Such a shift is found only when some frame lands clearly on the
  wrong side of a boundary (but see the next section for a slow-readout camera).

So a clean result does not prove alignment; an issue does prove misalignment.

The shift the checks above miss most: one stray edge after "ready"
-------------------------------------------------------------------
One stray trigger edge after the camera was reported ready (before shot 0's
first real trigger, or anywhere later) moves every later frame one slot late,
and the count stays N/N: the run's last real frame is the one never filed. The
stray frame is filed as shot 0's first image but arrived *after* ``t_start``, and
each frame pushed into the next shot's first slot is a dark frame, which arrives
just after that boundary -- none of it is early. What does change is the LAST
slot of each shot: it now holds that shot's light frame instead of its dark.

What does show it is where the long gaps fall. A shot's frames come within tens
of milliseconds of each other; shots are seconds apart. So in a good run the long
gap before each shot's frames is the one before its FIRST slot. After a stray
edge it moves to the gap before slot 1 (the first slot now holds the previous
shot's last frame); after a missed trigger, to the gap before the last slot. For
every shot after the first, the slot with the longest gap before it is found;
where that gap is decisive (at least ``GAP_DOMINANCE`` times the shot's next
longest) and is not before slot 0, the shot is counted, and ``k`` such shots out
of ``N`` give a NOTE ("possible one-slot shift"), never an issue:

* Timing alone decides it, not SHOT_COMPLETE, so a sequence that runs on after
  its last image, or a SHOT_COMPLETE stamped late behind another request, does
  not fake it. (2026-09-27: the earlier rule -- "the last slot must arrive after
  SHOT_COMPLETE" -- noted every shot of a correct Andor run, 83129, whose
  sequence runs 0.66 s past its dark frame.)
* A stalled image dispatcher stamps frames late: that lengthens one gap and
  shortens the next, so it does not make a gap dominate (tested for stalls of 2
  and 12 s). Were it ever to, the note names the shots, and a real shift runs on
  to the end or until a missed trigger undoes it.
* A sequence whose own frames are seconds apart within a shot (longer than the
  gap between shots) would be noted; no lab sequence does that today.

It cannot see a shift whose extra and missing frames cancel inside one shot, a
shift in shot 0 alone (no gap before its first frame to compare), or a camera
with one frame per shot. A clean result here proves nothing either.

``dark_after_shot_complete`` is accepted and ignored (the gap check runs for
every camera); ``readout_outlasts_rpc`` stays for callers.

Both lists of times must come from one monotonic clock in one process (liveOD
uses ``time.monotonic()``: QueryPerformanceCounter on Windows, 100 ns). The
causal gap a real frame has over its bound is at least the server's reply, the
kernel's RPC return, the trigger, the exposure and the readout -- milliseconds --
so ``margin_s`` only has to cover clock granularity; 5 ms is generous.

Pure: no Qt, no numpy, no I/O.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# A frame must precede its bound by more than this to count as early.
DEFAULT_MARGIN_S = 0.005
# A frame this long after its shot was reported complete is noted (never an issue).
DEFAULT_LATE_S = 1.0
# Camera types whose readout outlasts the SHOT_COMPLETE RPC (the Andor EMCCD's
# full-frame readout, ~18 ms, against ~3 ms). No longer used by the check itself.
SLOW_READOUT_CAMERA_TYPES = ("andor",)
# A shot counts as shifted only when its longest gap is at least this many times
# its next longest: a shot whose gaps are all alike says nothing.
GAP_DOMINANCE = 3.0


def readout_outlasts_rpc(camera_params) -> bool:
    """``dark_after_shot_complete`` for a camera: True for the camera types in
    SLOW_READOUT_CAMERA_TYPES, False for any other type and for None (a camera
    the caller could not look up)."""
    camera_type = getattr(camera_params, "camera_type", None)
    if isinstance(camera_type, bytes):
        camera_type = camera_type.decode(errors="replace")
    return camera_type in SLOW_READOUT_CAMERA_TYPES


@dataclass
class Assessment:
    """``issues``: proof that frames are not in their shots' slots (the run's images
    must not be read shot by shot). ``notes``: things that may or may not mean
    that, for a WARNING."""
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def assess(frame_t: Sequence[float], shot_t: Sequence[float], per_shot: int,
           t_start: Optional[float] = None, margin_s: float = DEFAULT_MARGIN_S,
           late_s: float = DEFAULT_LATE_S,
           dark_after_shot_complete: bool = False) -> Assessment:
    """Check frame arrival times against the shots' completion times.

    ``frame_t``: arrival time of each frame, in the order the frames were filed.
    ``shot_t``: when each shot was reported complete, in order.
    ``per_shot``: frames per shot. ``t_start``: when the experiment was told the
    camera was ready (None: shot 0's frames are not checked against a start).
    ``dark_after_shot_complete``: accepted and ignored; the one-slot-shift note
    (module docstring) comes from the gaps between frames, for every camera.
    """
    out = Assessment()
    per_shot = int(per_shot)
    frame_t = [float(t) for t in frame_t]
    shot_t = [float(t) for t in shot_t]
    if per_shot < 1 or not frame_t:
        return out
    if not shot_t:
        out.notes.append(f"{len(frame_t)} frame(s) but no shot was reported complete; "
                         f"frame alignment not checked")
        return out
    n_rep = len(shot_t)

    early = []      # (frame, shot, seconds before the bound, what the bound was)
    beyond = []     # frames in the slot of a shot that never started
    late = []       # (frame, shot, seconds after the shot was reported complete)
    after_last = []  # frames in the slot of the shot after the last one reported
    for i, t in enumerate(frame_t):
        s = i // per_shot
        if s == 0:
            lo, what = t_start, "the camera was reported ready"
        elif s - 1 < n_rep:
            lo, what = shot_t[s - 1], f"shot {s - 1} was reported complete"
        else:
            beyond.append((i, s))
            continue
        if lo is not None and t < lo - margin_s:
            early.append((i, s, lo - t, what))
        if s < n_rep:
            if t > shot_t[s] + late_s:
                late.append((i, s, t - shot_t[s]))
        else:
            after_last.append((i, s))

    if early:
        i, s, dt, what = early[0]
        out.issues.append(
            f"frame {i} is filed as shot {s}'s but arrived {dt:.3f} s before {what}, "
            f"so no trigger of shot {s} made it; an extra frame came earlier and the "
            f"frames from there on are in the wrong shots' slots "
            f"({len(early)} frame(s) arrived before their shot could start)")
    if beyond:
        i, s = beyond[0]
        out.issues.append(
            f"frame {i} is filed as shot {s}'s, but only {n_rep} shot(s) were reported "
            f"complete, so shot {s} never ran ({len(beyond)} frame(s) beyond the "
            f"reported shots)")
    if late:
        i, s, dt = late[0]
        out.notes.append(
            f"frame {i} (shot {s}) arrived {dt:.3f} s after shot {s} was reported "
            f"complete: a slow frame, or a frame of a later shot after a missed trigger "
            f"({len(late)} frame(s) more than {late_s:g} s late)")
    if after_last:
        i, s = after_last[0]
        out.notes.append(
            f"frame {i} is filed as shot {s}'s, the shot after the last one reported "
            f"complete ({len(after_last)} such frame(s))")
    note = _long_gap_off_slot_zero(frame_t, shot_t, per_shot)
    if note:
        out.notes.append(note)
    return out


def _long_gap_off_slot_zero(frame_t, shot_t, per_shot) -> str:
    """The one-slot-shift note (module docstring), or "" when every decisive shot
    has its long gap before slot 0. Shots 1.. that were reported complete and
    whose slots were all filled are checked."""
    if per_shot < 2:
        return ""
    n_filled = min(len(shot_t), len(frame_t) // per_shot)
    checked, shifted = [], []   # shot indices; (shot, slot, gap in s)
    for s in range(1, n_filled):
        gaps = [frame_t[s * per_shot + p] - frame_t[s * per_shot + p - 1]
                for p in range(per_shot)]
        order = sorted(range(per_shot), key=lambda p: gaps[p], reverse=True)
        top, second = gaps[order[0]], max(gaps[order[1]], 0.0)
        if top <= 0 or top < GAP_DOMINANCE * second:
            continue            # no decisive long gap in this shot: says nothing
        checked.append(s)
        if order[0] != 0:
            shifted.append((s, order[0], top))
    if not shifted:
        return ""
    first, slot, gap = shifted[0]
    last = shifted[-1][0]
    to_the_end = [s for s, _, _ in shifted] == [s for s in checked if s >= first]
    direction = ("one slot late: an extra frame (a stray trigger edge) came earlier"
                 if slot == 1 else
                 "one slot early: a trigger was missed earlier" if slot == per_shot - 1
                 else f"{slot} slot(s) off")
    return (f"possible one-slot shift: in {len(shifted)}/{len(checked)} shots the long gap "
            f"between shots falls before slot {slot}, not slot 0 (shots {first}..{last}"
            f"{', every shot from there to the end' if to_the_end else ''}; shot {first}'s gap "
            f"{gap:.3f} s) -- frames {direction}, or a stalled image dispatcher stamped them late")


def check(frame_t: Sequence[float], shot_t: Sequence[float], per_shot: int,
          t_start: Optional[float] = None, margin_s: float = DEFAULT_MARGIN_S) -> List[str]:
    """The unambiguous findings only (see ``assess``): an empty list means no
    proof of misalignment, not proof of alignment."""
    return assess(frame_t, shot_t, per_shot, t_start, margin_s).issues
