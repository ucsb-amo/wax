"""waxx.util.live_od.frame_alignment: what frame arrival times can and cannot
prove about frames sitting in their shots' slots. Pure; no Qt, no files.

Timeline used below, as a real run looks: the camera is reported ready at t=0;
shot k is reported complete at t = 10*(k+1); its three frames (atoms, light,
dark) come at the end of the shot, 100 ms and 50 ms before the shot is reported
complete and 2 ms after it (the dark frame's readout outlasts the cleanup).
"""
from waxx.util.live_od import frame_alignment as fa

PER_SHOT = 3


def shots(n):
    return [10.0 * (k + 1) for k in range(n)]


def good_frames(n_shots):
    out = []
    for k in range(n_shots):
        end = 10.0 * (k + 1)
        out += [end - 0.10, end - 0.05, end + 0.002]
    return out


def test_a_good_run_has_no_issues_and_no_notes():
    out = fa.assess(good_frames(4), shots(4), PER_SHOT, t_start=0.0)
    assert out.issues == [] and out.notes == []
    assert fa.check(good_frames(4), shots(4), PER_SHOT, 0.0) == []


def test_a_stray_frame_before_the_camera_was_ready_is_an_issue():
    frames = [-0.5] + good_frames(3)[:-1]     # stray first: every frame one slot late, count right
    issues = fa.check(frames, shots(3), PER_SHOT, t_start=0.0)
    assert len(issues) == 1
    assert "frame 0" in issues[0] and "camera was reported ready" in issues[0]


def test_frames_shifted_by_two_across_a_shot_boundary_are_an_issue():
    # two stray frames in shot 0: shot 0's light frame lands in shot 1's first slot,
    # and it arrived 50 ms before shot 0 was even reported complete
    frames = [5.0, 5.1] + good_frames(3)[:-2]
    issues = fa.check(frames, shots(3), PER_SHOT, t_start=0.0)
    assert len(issues) == 1
    assert "frame 3 is filed as shot 1's" in issues[0] and "shot 0 was reported complete" in issues[0]


def test_a_one_frame_shift_is_not_provable_from_times():
    """The frame pushed into the next shot's slot is a dark frame, which arrives
    right around the boundary: nothing is proved, and nothing is claimed."""
    frames = [5.0] + good_frames(3)[:-1]
    assert fa.check(frames, shots(3), PER_SHOT, t_start=0.0) == []


def test_late_frames_are_only_notes():
    frames = good_frames(3)
    frames[2] = 10.0 + 3.0                    # shot 0's dark frame 3 s late (a stalled thread?)
    frames[3:] = [t + 3.0 for t in frames[3:]]
    out = fa.assess(frames, shots(3), PER_SHOT, 0.0, late_s=1.0)
    assert out.issues == []
    assert len(out.notes) == 1 and "frame 2 (shot 0)" in out.notes[0]


def test_frames_beyond_the_reported_shots_are_an_issue():
    frames = good_frames(2) + [25.0, 25.1, 25.2, 35.0]   # 2 shots reported, 4 shots' worth of slots used
    out = fa.assess(frames, shots(2), PER_SHOT, 0.0)
    assert len(out.issues) == 1 and "only 2 shot(s) were reported complete" in out.issues[0]
    assert any("the shot after the last one reported" in n for n in out.notes)


def test_no_shot_reported_is_not_checked():
    out = fa.assess(good_frames(1), [], PER_SHOT, 0.0)
    assert out.issues == [] and "not checked" in out.notes[0]


def test_without_a_start_time_shot_zero_has_no_lower_bound():
    frames = [-5.0] + good_frames(2)[1:]
    assert fa.check(frames, shots(2), PER_SHOT, t_start=None) == []


def test_the_margin_only_absorbs_clock_granularity():
    frames = good_frames(2)
    frames[2] = 9.98
    frames[3] = 10.0                          # exactly at the bound: not provably early
    assert fa.check(frames, shots(2), PER_SHOT, 0.0) == []
    frames[3] = 10.0 - 0.01                   # 10 ms before the bound: early (margin 5 ms)
    assert fa.check(frames, shots(2), PER_SHOT, 0.0)


def test_degenerate_inputs():
    assert fa.check([], shots(2), PER_SHOT, 0.0) == []
    assert fa.check(good_frames(1), shots(1), 0, 0.0) == []


# ----------------------------------------------------------------------
# dark_after_shot_complete: a one-slot shift from a stray edge after "ready"
# ----------------------------------------------------------------------
# The reviewer's numbers: triggers 0/30/60 ms into the shot, the camera's
# readout 18 ms (Andor), the SHOT_COMPLETE RPC 3 ms after the last trigger.
# Shot k starts at 1 + 10*k s; the camera was reported ready at t = 0.

TRIGGERS_S = (0.0, 0.030, 0.060)
RPC_S = 0.003
N_SHOTS = 4


def shot_start(k):
    return 1.0 + 10.0 * k


def timed_shots(n=N_SHOTS, rpc_s=RPC_S):
    return [shot_start(k) + TRIGGERS_S[-1] + rpc_s for k in range(n)]


def timed_frames(n=N_SHOTS, readout_s=0.018):
    return [shot_start(k) + t + readout_s for k in range(n) for t in TRIGGERS_S]


def test_a_good_slow_readout_run_has_no_note():
    frames, shot_t = timed_frames(), timed_shots()
    assert frames[2] - shot_t[0] > 0.010                    # the dark arrives 15 ms after SHOT_COMPLETE
    out = fa.assess(frames, shot_t, PER_SHOT, t_start=0.0, dark_after_shot_complete=True)
    assert out.issues == [] and out.notes == []


def test_a_sequence_that_runs_on_after_its_last_image_is_not_noted():
    """Run 83129 (2026-09-26): a correct Andor run whose sequence runs 0.66 s past
    its dark frame, so every dark arrived 0.66 s BEFORE SHOT_COMPLETE (frames
    checked by eye: beam in atoms + light, none in dark, every shot). The old
    rule noted all 5 shots; the gaps say nothing is wrong."""
    frames = timed_frames(n=5)
    shot_t = [shot_start(k) + TRIGGERS_S[-1] + 0.018 + 0.66 for k in range(5)]
    out = fa.assess(frames, shot_t, PER_SHOT, t_start=0.0, dark_after_shot_complete=True)
    assert out.issues == [] and out.notes == []


def test_a_stray_edge_after_ready_is_noted_but_is_not_an_issue():
    # the stray edge between "ready" and shot 0: every frame one slot late, count N/N
    frames = [0.5] + timed_frames()[:-1]
    shot_t = timed_shots()
    out = fa.assess(frames, shot_t, PER_SHOT, t_start=0.0)
    assert out.issues == []                                  # a note only, never incomplete-marking
    assert fa.check(frames, shot_t, PER_SHOT, 0.0) == []
    assert len(out.notes) == 1
    note = out.notes[0]
    assert note.startswith("possible one-slot shift: in 3/3 shots the long gap between shots "
                           "falls before slot 1, not slot 0")
    assert "shots 1..3, every shot from there to the end" in note
    assert "one slot late" in note


def test_a_missed_trigger_is_noted_as_one_slot_early():
    # shot 1's atoms trigger missed: from there every frame one slot early
    good = timed_frames()
    frames = good[:3] + good[4:]
    out = fa.assess(frames, timed_shots(), PER_SHOT, 0.0)
    assert out.issues == []
    shift = [n for n in out.notes if n.startswith("possible one-slot shift")]
    assert len(shift) == 1 and "falls before slot 2" in shift[0]
    assert "one slot early" in shift[0]


def test_a_stray_edge_mid_run_is_noted_from_the_next_shot_on():
    # a stray 1 s before shot 2 lands in shot 2's first slot, where the long gap
    # still precedes slot 0; from shot 3 on the gap sits before slot 1
    good = timed_frames()
    frames = good[:6] + [shot_start(2) - 1.0] + good[6:-1]
    out = fa.assess(frames, timed_shots(), PER_SHOT, 0.0)
    assert out.issues == []
    assert len(out.notes) == 1 and "in 1/3 shots" in out.notes[0]
    assert "shots 3..3, every shot from there to the end" in out.notes[0]


def test_a_late_recorded_shot_complete_is_not_noted():
    """SHOT_COMPLETE times no longer matter to the note: a shot whose message waited
    50 ms in the server's queue is not noted (the old rule noted it)."""
    shot_t = timed_shots()
    shot_t[1] += 0.050
    out = fa.assess(timed_frames(), shot_t, PER_SHOT, 0.0, dark_after_shot_complete=True)
    assert out.issues == [] and out.notes == []


def test_a_fast_readout_camera_is_checked_too():
    # Basler-like: 1 ms readout. A good run is quiet whatever the RPC time; a
    # shifted one is noted (the old rule was blind here).
    for rpc_s in (0.003, 0.010):
        frames, shot_t = timed_frames(readout_s=0.001), timed_shots(rpc_s=rpc_s)
        good = fa.assess(frames, shot_t, PER_SHOT, 0.0)
        assert good.issues == [] and good.notes == []
    # (with a 10 ms RPC the shifted run is already an issue: each moved dark frame
    # arrives before the shot it is filed under could start)
    frames, shot_t = timed_frames(readout_s=0.001), timed_shots(rpc_s=0.003)
    shifted = fa.assess([0.5] + frames[:-1], shot_t, PER_SHOT, 0.0)
    assert shifted.issues == [] and "in 3/3 shots" in shifted.notes[0]


def test_a_stalled_dispatcher_does_not_fake_a_shift():
    # shot 2's light + dark stamped late by 2 s, or by 12 s (longer than a shot):
    # the stall lengthens one gap and shortens the next, so no gap dominates
    for stall_s in (2.0, 12.0):
        frames = timed_frames()
        frames[7:9] = [t + stall_s for t in frames[7:9]]
        out = fa.assess(frames, timed_shots(), PER_SHOT, 0.0, late_s=100.0)
        assert out.issues == [] and out.notes == []


def test_shots_without_a_decisive_gap_and_single_frame_cameras_say_nothing():
    evenly = [float(i) for i in range(12)]                  # every gap 1 s: nothing decisive
    assert fa.assess(evenly, timed_shots(), PER_SHOT, None).notes == []
    assert fa.assess(timed_frames()[::3], timed_shots(), 1, 0.0).notes == []


def test_only_shots_reported_and_filled_are_counted():
    frames = [0.5] + timed_frames()[:-1]
    out = fa.assess(frames[:7], timed_shots()[:2], PER_SHOT, 0.0, dark_after_shot_complete=True)
    assert out.issues == []
    assert any("in 1/1 shots" in n for n in out.notes)


def test_which_cameras_read_out_slower_than_the_rpc():
    from types import SimpleNamespace as NS
    assert fa.readout_outlasts_rpc(NS(camera_type="andor"))
    assert fa.readout_outlasts_rpc(NS(camera_type=b"andor"))
    assert not fa.readout_outlasts_rpc(NS(camera_type="basler"))
    assert not fa.readout_outlasts_rpc(NS(camera_type="apd"))
    assert not fa.readout_outlasts_rpc(None)
