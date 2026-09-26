"""The tweezer AWG as a connection the monitor server holds between runs.

:class:`TweezerAwgDriver` is the driver its agent process runs (see
:mod:`waxx.util.device_state.connection_agent` and
:mod:`waxx.util.device_state.connections`).  It opens and closes the card
exactly as a run does -- ``TweezerController.awg_init`` (DDS mode, trigger on
ext0, its DDS reset leaving no tones) and the bounded
``AwgConnection.close`` (stop with STOPDMA, spcm_vClose in a worker thread) --
and writes static tones with :func:`write_static_tones`.

New tones take effect on the next trigger.  The trigger is the ARTIQ TTL
``awg_trg_ttl`` (the Composite tab's "Apply traps" op pulses it after the
write); :meth:`TweezerAwgDriver.force_trigger` is the card's software
trigger, not yet verified on this card with its ext0-only trigger mask (see
``kexp/experiments/tools/awg_force_trigger_check.py``).
"""

from __future__ import annotations

#: How long an open waits while another connection has the card (a run
#: waits 30 s).  The monitor server opens on its own thread, so this can be
#: longer than the monitor loop could afford.
T_OPEN_WAIT_IN_USE = 10.
#: The driver's own close limit; the server kills the agent a little later.
T_CLOSE = 2.5


def write_static_tones(tw, rows) -> int:
    """Program static tones on ``tw`` (a TweezerController with the card
    open): one per ``[frequency, amplitude]`` row, and zero any tone this
    wrote before that is no longer listed (set_static_tweezers only writes
    the tones it is given).  Takes effect on the next AWG trigger.  Returns
    the number of tones."""
    rows = [[float(r[0]), float(r[1])] for r in rows]
    freqs = [r[0] for r in rows]
    amps = [r[1] for r in rows]
    if sum(amps) > 1. + 1e-9:
        raise ValueError(f"amplitudes sum to {sum(amps):.3f} > 1")
    if any(a < 0. for a in amps):
        raise ValueError("negative amplitude")
    n_cores = len(getattr(tw, "core_list", ()) or ()) or len(rows)
    if len(rows) > n_cores:
        raise ValueError(f"{len(rows)} tones, but the card has {n_cores} DDS cores")
    n_prev = int(getattr(tw, "_panel_n_tones", 0))
    if freqs:
        if sum(amps) > 0.:
            tw.set_static_tweezers(freqs, amps)
        else:
            # compute_tweezer_phases divides by the total amplitude.
            tw.set_static_tweezers(freqs, amps, [0.] * len(freqs))
    if n_prev > len(freqs):
        for idx in range(len(freqs), n_prev):
            tw.dds[idx].amp(0.)
        tw.dds.exec_at_trg()
        tw.dds.write()
    tw._panel_n_tones = len(freqs)
    tw._panel_traps = rows
    return len(rows)


class TweezerAwgDriver:
    """The AWG connection's driver (runs in the agent process)."""

    COMMANDS = ("write_traps", "force_trigger")

    def __init__(self, awg_ip: str, t_wait_in_use: float = T_OPEN_WAIT_IN_USE,
                 close_timeout: float = T_CLOSE, controller=None):
        if controller is None:
            from waxx.control.tweezer.spectrum_DDS_tweezer import TweezerController  # noqa: PLC0415
            controller = TweezerController(awg_ip=awg_ip)
        self.tw = controller
        self.awg_ip = str(awg_ip)
        self.t_wait_in_use = float(t_wait_in_use)
        self.close_timeout = float(close_timeout)
        self._forget_tones()

    def _forget_tones(self) -> None:
        self.tw._panel_n_tones = 0
        self.tw._panel_traps = []

    def open(self) -> None:
        self.tw.awg_init(t_wait_in_use=self.t_wait_in_use)
        self._forget_tones()

    def close(self) -> bool:
        closed = self.tw.close(timeout=self.close_timeout)
        self._forget_tones()
        return closed is not False

    def is_open(self) -> bool:
        return getattr(self.tw, "card", None) is not None

    def detail(self) -> str:
        traps = getattr(self.tw, "_panel_traps", []) or []
        host = self.awg_ip.split("::")[1] if "::" in self.awg_ip else self.awg_ip
        tones = (f"{len(traps)} tone(s): " + ", ".join(f"{f / 1e6:.3f}" for f, _ in traps)
                 + " MHz") if traps else "no tones loaded"
        return f"{host} · {tones}" if host else tones

    def write_traps(self, rows) -> dict:
        if not self.is_open():
            raise RuntimeError("the AWG is not open")
        return {"tones": write_static_tones(self.tw, rows)}

    def force_trigger(self) -> None:
        """The card's software trigger (M2CMD_CARD_FORCETRIGGER).  Unverified
        with this card's DDS setup; see the module docstring."""
        if not self.is_open():
            raise RuntimeError("the AWG is not open")
        import spcm  # noqa: PLC0415
        self.tw.card.cmd(spcm.M2CMD_CARD_FORCETRIGGER)
