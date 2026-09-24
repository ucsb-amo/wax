"""LinkLatch: trip once, skip during the cooldown, retry after it, clear on
success. Prints exactly once per outage edge."""

from waxx.util.link_latch import LinkLatch, LinkDownError


class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t


def test_fresh_latch_is_up():
    latch = LinkLatch("dev", retry_after=10., clock=Clock())
    assert not latch.down
    assert not latch.should_skip()
    latch.raise_if_skipping()   # no raise


def test_trip_skips_until_cooldown_elapses():
    clock = Clock()
    latch = LinkLatch("dev", retry_after=10., clock=clock)
    latch.trip(OSError("unreachable"))
    assert latch.down
    assert latch.should_skip()
    clock.t += 9.9
    assert latch.should_skip()
    clock.t += 0.2
    assert not latch.should_skip()     # retry allowed
    assert latch.down                  # ...but still down until a success


def test_raise_if_skipping():
    clock = Clock()
    latch = LinkLatch("dev", retry_after=10., clock=clock)
    latch.trip(OSError("x"))
    try:
        latch.raise_if_skipping()
    except LinkDownError as e:
        assert "dev" in str(e)
    else:
        raise AssertionError("expected LinkDownError")
    assert issubclass(LinkDownError, ConnectionError)


def test_retrip_rearms_cooldown_without_reprinting(capsys):
    clock = Clock()
    latch = LinkLatch("dev", retry_after=10., clock=clock)
    latch.trip(OSError("first"))
    clock.t += 11.
    assert not latch.should_skip()
    latch.trip(OSError("second"))
    assert latch.should_skip()
    assert latch.last_error.args == ("second",)
    out = capsys.readouterr().out
    assert out.count("link down") == 1


def test_clear_prints_once_and_resets(capsys):
    clock = Clock()
    latch = LinkLatch("dev", retry_after=10., clock=clock)
    latch.clear()                       # up -> up: silent
    assert capsys.readouterr().out == ""
    latch.trip(OSError("x"))
    latch.clear()
    latch.clear()
    out = capsys.readouterr().out
    assert out.count("link back") == 1
    assert not latch.down
    assert latch.last_error is None
