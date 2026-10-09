"""waxx.calibration -- calibration emit, ledger and write-back.

An experiment says which params it calibrates; at the end of the run (after
liveOD has saved it) each one is analysed, checked, recorded in the ledger and,
when allowed, written into the params source file -- the old line commented
out in place, the new one below it tagged with the run id and date. Write-backs
stay uncommitted in git, like hand edits; ``kcal revert <key>`` undoes one.

Usage, in an experiment's prepare()::

    self.xvar('t_raman_pulse', np.linspace(0., 20.e-6, 21))
    self.calibrates('t_raman_pi_pulse', analysis='rabi_pi_time')   # write_back=True
    self.finish_prepare(shuffle=True)

What the terminal prints after the run is saved (values for illustration)::

    [cal] t_raman_pi_pulse = 6.612e-06 +/- 2.1e-08 s (#85600, 63 shots, 0 excluded; was 6.6403e-06, -0.43 %)
    [cal] applied: ...\\kexp\\config\\expt_params.py:312 (kcal revert t_raman_pi_pulse undoes it)
    [cal]   - self.t_raman_pi_pulse = 6.6403e-06 #85412, 2026-10-07
    [cal]   + self.t_raman_pi_pulse = 6.6120e-06 #85600, 2026-10-09

or, when it is not written back::

    [cal] not applied: flagged -- change from the value in use +12.000% exceeds the policy limit +/-5.000%. A flagged result is never written back.
    [cal] not applied: vetoed by WAXX_CAL_NO_WRITE_BACK; to apply: kcal apply t_raman_pi_pulse --run 85600

Options: ``write_back=False`` records without writing; ``allow_no_unc=True``
accepts a result without an uncertainty (written with its full repr);
``opts={...}`` goes to the analysis. A submitter vetoes every write-back of a
run with ``WAXX_CAL_NO_WRITE_BACK=1``. From analyze(), after ``self.end(...)``,
``self.emit_calibration(key, value, unc, unit=..., n_used=..., method=...)``
records a number the experiment computed itself (``write_back=False`` unless
given).

Pieces: ``record`` (CalResult, evaluate -- the flag rules), ``policy`` (per-key
thresholds; none = no hard checks), ``precision`` (how many digits are
written), ``writeback`` (find / apply / revert, verified), ``ledger``
(append-only jsonl + one JSON per key and run), ``analysis`` (the
``calibrate(ad, key, **opts)`` contract, registry, 30 s budget), ``emit`` (the
end-of-run pipeline), ``config`` (``CalibrationConfig``, filled by the machine)
and ``cli`` (``kcal``; config from ``--config module:attr`` or
``$WAXX_CALIBRATION_CONFIG``).

Not here yet (phase 2): refreshing a waiting run's params at GO, and the extra
host->kernel params sync at the top of init_kernel.
"""

from waxx.calibration.config import CalibrationConfig, load_config
from waxx.calibration.record import CalResult, evaluate

__all__ = ["CalResult", "CalibrationConfig", "evaluate", "load_config"]
