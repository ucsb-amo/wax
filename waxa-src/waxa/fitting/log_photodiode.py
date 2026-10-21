"""Calibration fit for a logarithmic photodiode: V = v_ref + k * log10(P / p_ref).

Points near the PD's floor (dark current, stray light: the output flattens) and near its top-end
compression are not fitted. The cuts are given by the caller in volts, and every excluded point is
returned with its reason, so the exclusions are stated, never silent. Residuals are also reported as
fractional power errors, which is how a residual matters to an intensity lock.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LogPhotodiodeFit:
    k: float                 # V per decade
    k_err: float
    v_ref: float             # V at p_ref
    v_ref_err: float
    p_ref: float             # reference power (definition, not fitted)
    n_used: int
    excluded: list = field(default_factory=list)   # (index, power, volts, reason)
    residual_rms_v: float = np.nan
    residual_max_frac: float = np.nan              # max |P_fit / P - 1| over used points

    def power_to_v(self, p):
        return self.v_ref + self.k * np.log10(np.asarray(p, dtype=float) / self.p_ref)

    def v_to_power(self, v):
        return self.p_ref * 10 ** ((np.asarray(v, dtype=float) - self.v_ref) / self.k)

    def summary(self) -> str:
        lines = [f"k = {self.k:.6g} +/- {self.k_err:.2g} V/decade, "
                 f"v_ref = {self.v_ref:.6g} +/- {self.v_ref_err:.2g} V at p_ref = {self.p_ref:.4g}",
                 f"{self.n_used} points fitted, {len(self.excluded)} excluded; "
                 f"residual rms {self.residual_rms_v * 1e3:.3g} mV, max {self.residual_max_frac * 100:.3g} % in power"]
        for idx, p, v, reason in self.excluded:
            lines.append(f"  excluded #{idx}: P = {p:.4g}, V = {v:.4g} ({reason})")
        return "\n".join(lines)


def fit_log_photodiode(power, volts, p_ref, v_floor_cut=-np.inf, v_sat_cut=np.inf) -> LogPhotodiodeFit:
    """Fit V = v_ref + k log10(P / p_ref) by least squares on points with v_floor_cut < V < v_sat_cut.

    power: reference power for each point (any unit; p_ref in the same unit), > 0.
    volts: log-PD reading for each point.
    v_floor_cut: points at or below are excluded (the floor region; e.g. dark reading + margin).
    v_sat_cut: points at or above are excluded (compression).
    Non-finite and non-positive-power points are excluded too. Uncertainties are from the fit
    covariance scaled by the residual variance (statistical only).
    """
    p = np.asarray(power, dtype=float).ravel()
    v = np.asarray(volts, dtype=float).ravel()
    if p.shape != v.shape:
        raise ValueError(f"power and volts differ in length: {p.size} vs {v.size}")

    excluded = []
    use = np.ones(p.size, dtype=bool)
    for i in range(p.size):
        reason = None
        if not (np.isfinite(p[i]) and np.isfinite(v[i])):
            reason = "non-finite"
        elif p[i] <= 0:
            reason = "power <= 0"
        elif v[i] <= v_floor_cut:
            reason = f"V <= floor cut {v_floor_cut:.4g}"
        elif v[i] >= v_sat_cut:
            reason = f"V >= saturation cut {v_sat_cut:.4g}"
        if reason is not None:
            use[i] = False
            excluded.append((i, p[i], v[i], reason))

    n = int(use.sum())
    if n < 3:
        raise ValueError(f"only {n} points inside the cuts; need at least 3 (excluded {len(excluded)})")

    x = np.log10(p[use] / p_ref)
    y = v[use]
    (k, v0), cov = np.polyfit(x, y, 1, cov=True)
    resid = y - (v0 + k * x)
    p_fit = p_ref * 10 ** ((y - v0) / k)
    return LogPhotodiodeFit(
        k=float(k), k_err=float(np.sqrt(cov[0, 0])),
        v_ref=float(v0), v_ref_err=float(np.sqrt(cov[1, 1])),
        p_ref=float(p_ref), n_used=n, excluded=excluded,
        residual_rms_v=float(np.sqrt(np.mean(resid ** 2))),
        residual_max_frac=float(np.max(np.abs(p_fit / p[use] - 1))),
    )
