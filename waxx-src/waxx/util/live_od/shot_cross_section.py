"""Per-shot absorption cross section for liveOD's atom numbers.

liveOD analyses a shot when its images arrive, but which cross section applies
can depend on conditions only known once the experiment has finished the shot
(on the K machine: the field at imaging, recorded as the outer-coil current at
the first camera trigger). The experiment sends what each shot recorded along
with SHOT_COMPLETE as ``shot_conditions``. So the two halves arrive separately,
in either order, and are paired here by shot index. Atom numbers are emitted
once both are in.

The rule is the lab's: ``LiveODConfig.cross_section_for_shot``, a function
``shot_conditions -> (sigma_m2, source_tag)``. ``cross_section_for_shot`` below is
the analysis's own rule (``waxa.calibrations.cross_section``), which kexp passes,
so that liveOD and atomdata agree shot for shot; every emitted dict carries the
value used and its source tag. With no rule configured the atom-number entries
stay integrated OD x area. A shot whose conditions never arrive (an experiment
process that predates ``shot_conditions``) is emitted with whatever the rule
returns for "no conditions" when the next shot is analysed.

No Qt in this module, so it can be tested without a GUI.
"""

from waxa.calibrations.cross_section import cross_section_from_outer_current, I_OUTER_KEY

# Scalars that are an integrated OD times an area and become an atom number on
# division by the cross section.
ATOM_NUMBER_KEYS = ('atom_number', 'atom_number_fit_area_x', 'atom_number_fit_area_y')


def cross_section_for_shot(shot_conditions):
    """``(sigma_m2, source_tag)`` for one shot from its recorded conditions
    (a dict, possibly empty or None)."""
    i_outer = None
    if shot_conditions:
        i_outer = shot_conditions.get(I_OUTER_KEY)
    sigma, source = cross_section_from_outer_current(i_outer)
    return float(sigma), str(source)


class ShotCrossSectionPairing:
    """Pairs each analysed shot with the conditions the experiment reports for it."""

    def __init__(self, cross_section_for_shot=None):
        """``cross_section_for_shot``: shot_conditions (dict or None) ->
        (sigma_m2, source_tag). None: nothing is divided."""
        self._rule = cross_section_for_shot
        self.reset()

    def reset(self):
        """New run: shot indices start again from zero."""
        self._analysed = {}     # shot_idx -> (scalars with un-divided atom numbers, calibrated)
        self._conditions = {}   # shot_idx -> (shot_conditions dict, xvar_values dict)

    def add_analysis(self, shot_idx, scalars, calibrated):
        """The images of ``shot_idx`` have been analysed.

        ``scalars[k]`` for k in ATOM_NUMBER_KEYS hold OD x area (m^2), not yet
        divided. ``calibrated`` is False when there is no pixel calibration: the
        numbers are then plain integrated OD and are left undivided, as before.
        Returns the scalar dicts that are now complete, oldest first.
        """
        done = []
        # An older shot still waiting was never reported by the experiment.
        for idx in sorted(k for k in self._analysed if k < shot_idx):
            done.append(self._finish(idx))
        self._analysed[shot_idx] = (scalars, calibrated)
        if shot_idx in self._conditions:
            done.append(self._finish(shot_idx))
        return done

    def add_conditions(self, shot_idx, shot_conditions, xvar_values=None):
        """The experiment has reported ``shot_idx`` complete. Returns the scalar
        dicts that are now complete."""
        self._conditions[shot_idx] = (dict(shot_conditions or {}), dict(xvar_values or {}))
        # conditions for shots that were never analysed (no camera) must not pile up
        for idx in [k for k in self._conditions if k < shot_idx and k not in self._analysed]:
            del self._conditions[idx]
        if shot_idx in self._analysed:
            return [self._finish(shot_idx)]
        return []

    def _finish(self, shot_idx):
        scalars, calibrated = self._analysed.pop(shot_idx)
        conditions, xvar_values = self._conditions.pop(shot_idx, (None, None))
        if self._rule is None:
            calibrated = False
            no_rule_tag = 'no-cross-section-rule'
        else:
            sigma, source = self._rule(conditions)
            sigma, source = float(sigma), str(source)
            no_rule_tag = 'uncalibrated-integrated-od'
        if calibrated:
            for key in ATOM_NUMBER_KEYS:
                if key in scalars:
                    scalars[key] = scalars[key] / sigma     # NaN stays NaN
            scalars['atom_cross_section_m2'] = sigma
            scalars['atom_cross_section_source'] = source
        else:
            scalars['atom_cross_section_m2'] = float('nan')
            scalars['atom_cross_section_source'] = no_rule_tag
        if xvar_values:
            # this shot's own values, not whatever the latest SHOT_COMPLETE said
            scalars['xvar_values'] = xvar_values
        return scalars
