"""The run's terminal lines at each verbosity: what is folded into one
summary line, and what always prints whatever the level."""

from types import SimpleNamespace

import pytest

from waxx.base.expt import Expt
from waxx.util import console


def _stream(ok, requested, discarded=0, complete=True):
    return SimpleNamespace(tally={"ok": ok, "requested": requested,
                                  "discarded": discarded, "complete": complete})


@pytest.fixture
def level(monkeypatch):
    def set_(lv):
        monkeypatch.setattr(console, "_level", lv)
    return set_


def test_one_line_for_all_complete_streams(capsys, level):
    level(console.NORMAL)
    Expt._camera_streams_summary([_stream(10, 10, 5), _stream(5, 5, 10), _stream(5, 5)])
    out = capsys.readouterr().out
    assert out.strip() == ("[cams] 20/20 frames captured in 3 stream(s) "
                           "(15 sibling-edge frames discarded)")


def test_the_summary_is_hidden_at_quiet(capsys, level):
    level(console.QUIET)
    Expt._camera_streams_summary([_stream(5, 5)])
    assert capsys.readouterr().out == ""


def test_a_stream_with_missing_frames_is_said_at_every_level(capsys, level):
    level(console.QUIET)
    Expt._camera_streams_summary([_stream(5, 5), _stream(3, 5, complete=False)])
    out = capsys.readouterr().out
    assert "8/10 frames" in out and "!! 1 stream(s) with missing frames" in out


def test_placeholder_and_unconnected_streams_are_not_counted(capsys, level):
    level(console.NORMAL)
    Expt._camera_streams_summary([SimpleNamespace(), _stream(5, 5)])
    assert "5/5 frames captured in 1 stream(s)" in capsys.readouterr().out
    Expt._camera_streams_summary([SimpleNamespace()])
    assert capsys.readouterr().out == ""
