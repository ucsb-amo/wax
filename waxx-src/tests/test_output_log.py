"""The output kept for the Sequences tab's logs (waxx.util.device_state.output_log).
Pure Python."""
from waxx.util.device_state.output_log import OutputLog


def test_lines_are_numbered_and_asked_for_after_the_last_one_seen():
    log = OutputLog()
    assert log.since(0) == {"lines": [], "first": 1, "next": 1, "more": False}
    for i in range(5):
        log.append(f"line {i}")
    reply = log.since(0)
    assert reply["lines"] == [f"line {i}" for i in range(5)]
    assert (reply["first"], reply["next"], reply["more"]) == (1, 6, False)
    assert log.since(3)["lines"] == ["line 3", "line 4"]
    nothing = log.since(5)
    assert nothing["lines"] == [] and nothing["first"] == nothing["next"] == 6


def test_dropped_lines_show_as_a_gap():
    log = OutputLog(maxlen=3)
    for i in range(10):
        log.append(str(i))
    reply = log.since(2)
    assert reply["lines"] == ["7", "8", "9"]
    assert reply["first"] == 8                  # 3..7 were dropped: first > after + 1


def test_a_reply_is_capped_and_says_more_is_waiting():
    log = OutputLog()
    for i in range(7):
        log.append(str(i))
    reply = log.since(0, limit=3)
    assert reply["lines"] == ["0", "1", "2"] and reply["next"] == 4 and reply["more"]
    reply = log.since(reply["next"] - 1, limit=3)
    assert reply["lines"] == ["3", "4", "5"] and reply["more"]
    reply = log.since(reply["next"] - 1, limit=3)
    assert reply["lines"] == ["6"] and not reply["more"]


def test_a_server_restart_shows_as_next_at_or_below_what_was_seen():
    fresh = OutputLog()
    fresh.append("first line after the restart")
    assert fresh.since(40)["next"] <= 40


def test_mark_is_a_timestamped_separator_and_bad_after_counts_as_zero():
    log = OutputLog()
    log.mark("1st run of the loop: auto_tof", clock=lambda: 0.0)
    line = log.since("garbage")["lines"][0]
    assert line.startswith("── ") and line.endswith(" 1st run of the loop: auto_tof ──")
