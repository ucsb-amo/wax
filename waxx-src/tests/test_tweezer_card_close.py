"""The tweezer AWG connection (waxx.control.tweezer.awg_connection, the parent
of TweezerController): connect() retries while another connection holds the
card; close() stops the card with STOPDMA, closes the handle, and never waits
longer than its timeout for the driver (spcm_vClose once hung for good, run
83101, and the run never reached end()).

The spcm card is replaced by a fake, so nothing touches the driver or the
network. The controller is made with __new__: connect(), close() and
reset_awg() only use the ``card`` / ``dds`` / ``_awg_ip`` attributes.
"""

import threading
import time

import pytest
import spcm

from waxx.control.tweezer import awg_connection as ac
from waxx.control.tweezer import spectrum_DDS_tweezer as sdt
from waxx.control.tweezer.spectrum_DDS_tweezer import TweezerController

ADDRESS = "TCPIP::192.0.2.1::inst0::INSTR"   # documentation range, never reached


class FakeCard:
    """Stands in for spcm.Card. ``hold_stop`` / ``hold_close`` are Events the
    matching call waits on (a driver call that does not return)."""

    def __init__(self, reject_stopdma=False, hold_stop=None, hold_close=None,
                 close_delay=0.):
        self.calls = []
        self._handle = object()
        self._closed = False
        self.reject_stopdma = reject_stopdma
        self.hold_stop = hold_stop
        self.hold_close = hold_close
        self.close_delay = close_delay

    def stop(self, *flags):
        # like spcm's Device._check_closed: no commands on a card marked closed
        if self._closed:
            raise spcm.SpcmException(text="The connection to the card has been closed.")
        self.calls.append(("stop", flags))
        if self.hold_stop is not None:
            self.hold_stop.wait()
        if flags and self.reject_stopdma:
            raise spcm.SpcmException(text="flag combination not supported")

    def handle(self):
        return self._handle

    def close(self, handle):
        self.calls.append(("close", handle))
        if self.close_delay:
            time.sleep(self.close_delay)
        if self.hold_close is not None:
            self.hold_close.wait()


class FakeDDS:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def controller(card):
    tw = TweezerController.__new__(TweezerController)
    tw._awg_ip = ADDRESS
    tw.card = card
    return tw


# --- close --------------------------------------------------------------------

def test_close_stops_dma_then_closes_the_handle():
    card = FakeCard()
    handle = card.handle()
    tw = controller(card)

    assert tw.close() is True

    assert card.calls == [("stop", (spcm.M2CMD_DATA_STOPDMA,)), ("close", handle)]
    assert tw.card is None
    assert card._closed is True      # Device.__del__ will not close it again
    assert card._handle is None


def test_close_without_a_card_does_nothing():
    tw = controller(None)
    assert tw.close() is True


def test_rejected_stopdma_falls_back_to_a_plain_stop(capsys):
    card = FakeCard(reject_stopdma=True)
    tw = controller(card)

    assert tw.close() is True

    assert [c[0] for c in card.calls] == ["stop", "stop", "close"]
    assert card.calls[1] == ("stop", ())
    assert "tweezer awg: stop with STOPDMA failed" in capsys.readouterr().out


def test_a_close_that_never_returns_is_given_up_on(capsys):
    release = threading.Event()
    card = FakeCard(hold_close=release)
    tw = controller(card)
    try:
        t0 = time.monotonic()
        assert tw.close(timeout=0.2) is False
        assert time.monotonic() - t0 < 2.

        out = capsys.readouterr().out
        assert "the card was stopped" in out
        assert "spcm_vClose" in out and "did not return within 0.2 s" in out
        # let go of: no later close or __del__ touches this card again
        assert tw.card is None
        assert card._closed is True
        assert tw.close() is True
    finally:
        release.set()


def test_a_stop_that_never_returns_says_the_card_may_still_run(capsys):
    release = threading.Event()
    card = FakeCard(hold_stop=release)
    tw = controller(card)
    try:
        assert tw.close(timeout=0.2) is False
        assert "may still be running" in capsys.readouterr().out
    finally:
        release.set()


def test_a_slow_close_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(ac, "T_AWG_CLOSE_SLOW", 0.05)
    tw = controller(FakeCard(close_delay=0.15))

    assert tw.close(timeout=2.) is True
    assert "closing the card took" in capsys.readouterr().out


def test_exit_handler_uses_the_exit_timeout(monkeypatch, capsys):
    monkeypatch.setattr(ac, "T_AWG_EXIT_CLOSE_TIMEOUT", 0.2)
    release = threading.Event()
    tw = controller(FakeCard(hold_close=release))
    try:
        t0 = time.monotonic()
        tw._close_at_exit()
        assert time.monotonic() - t0 < 2.
        assert "at exit" in capsys.readouterr().out
    finally:
        release.set()


def test_reset_awg_resets_the_dds_then_closes(monkeypatch):
    monkeypatch.setattr(ac, "T_AWG_CLOSE_TIMEOUT", 0.2)
    release = threading.Event()
    card = FakeCard(hold_close=release)
    tw = controller(card)
    tw.dds = FakeDDS()
    try:
        t0 = time.monotonic()
        tw.reset_awg()              # the post_scan call: must come back
        assert time.monotonic() - t0 < 2.
        assert tw.dds.resets == 1
        assert [c[0] for c in card.calls] == ["stop", "close"]
        assert tw.card is None
    finally:
        release.set()


# --- connect ------------------------------------------------------------------

@pytest.fixture
def no_exit_handlers(monkeypatch):
    registered = []
    monkeypatch.setattr(ac.atexit, "register", registered.append)
    return registered


def test_connect_opens_sets_up_and_registers_the_exit_close(no_exit_handlers):
    card = FakeCard()
    tw = controller(None)
    tw._open_card = lambda: card
    set_up = []

    tw.connect(lambda: set_up.append(tw.card))

    assert set_up == [card] and tw.card is card
    assert no_exit_handlers == [tw._close_at_exit]


def test_connect_waits_while_another_connection_holds_the_card(monkeypatch, capsys,
                                                               no_exit_handlers):
    monkeypatch.setattr(ac, "T_AWG_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(ac, "awg_holder_text", lambda ip: "held by 192.0.2.7")
    card = FakeCard()
    attempts = []

    def opener():
        attempts.append(1)
        if len(attempts) < 3:
            raise spcm.SpcmException(text="open: card is already in use by another application")
        return card

    tw = controller(None)
    tw._open_card = opener
    tw.connect(lambda: None, t_wait_in_use=5.)

    out = capsys.readouterr().out
    assert tw.card is card and len(attempts) == 3
    assert out.count("tweezer awg is in use (held by 192.0.2.7)") == 1   # once per holder
    assert "tweezer awg connected (attempt 3)." in out


def test_connect_gives_up_on_a_held_card_and_says_where_to_look(monkeypatch,
                                                                no_exit_handlers):
    monkeypatch.setattr(ac, "T_AWG_RETRY_INTERVAL", 0.01)
    monkeypatch.setattr(ac, "awg_holder_text", lambda ip: "held by 192.0.2.7")

    def opener():
        raise spcm.SpcmException(text="card is already in use")

    tw = controller(None)
    tw._open_card = opener
    with pytest.raises(RuntimeError) as err:
        tw.connect(lambda: None, t_wait_in_use=0.05)

    msg = str(err.value)
    assert msg.startswith("tweezer awg init failed: ")
    assert "(held by 192.0.2.7) -- still in use after 0.05 s. Close whatever has it open" in msg


def test_a_setup_error_closes_the_card_and_is_not_retried(no_exit_handlers):
    card = FakeCard()
    opens = []
    tw = controller(None)
    tw._open_card = lambda: (opens.append(1), card)[1]

    def setup():
        raise spcm.SpcmException(text="card mode not supported")

    with pytest.raises(RuntimeError, match="tweezer awg init failed: card mode not supported"):
        tw.connect(setup)

    assert len(opens) == 1
    assert tw.card is None and card._closed is True
    assert no_exit_handlers == []


def test_old_names_still_import_from_the_tweezer_module():
    assert issubclass(TweezerController, ac.AwgConnection)
    for name in ("AwgConnectionError", "awg_error_text", "awg_driver_error_text",
                 "awg_holder", "awg_holder_text", "is_retryable_awg_error",
                 "is_awg_in_use_error", "T_AWG_IN_USE_WAIT", "T_AWG_RETRY_INTERVAL",
                 "N_AWG_RETRIES", "T_AWG_CLOSE_TIMEOUT", "T_AWG_EXIT_CLOSE_TIMEOUT",
                 "T_AWG_CLOSE_SLOW"):
        assert getattr(sdt, name) is getattr(ac, name), name
