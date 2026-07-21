"""AtomdataVault: concatenate several single-axis atomdata runs into one
analysis object.

Intended use case: a 1-D scan was broken across multiple runs to keep per-file
sizes manageable, or an *experiment builder* launched many runs that each scan
the **same xvar over a different range**. AtomdataVault stitches the chunks back
together so the combined dataset can be analyzed with the usual atomdata
interface (``vault.atom_number``, ``vault.od``, ``vault.fit_sd_x``,
``vault.data.*``, etc.) without ever re-saving anything to disk.

Key capabilities:
    * Ragged-repeat-aware statistics. ``vault.avg`` / ``vault.std`` / ``vault.sem``
      group by unique xvar value, so overlapping ranges (where some points are
      sampled more often than others) average correctly. SEM uses each point's
      own repeat count. See ``collapse_to_unique`` to bake the grouping in.
    * Per-shot provenance. ``vault.shot_run_id`` records which source run each
      shot came from (carried through the internal sort), enabling drift plots
      coloured by run, ``shots_from_run``, and ``drop_runs``.
    * Source-run access. ``vault.run_info.run_id`` lists all source run IDs, and
      ``vault.atomdata(run_id)`` returns an already-loaded source run object.
    * Memory controls for large jobs: ``auto_lite_threshold`` and
      ``drop_raw_images`` keep many-run loads tractable.
    * Builder-aware discovery: ``AtomdataVault.from_run_range`` /
      ``from_builder`` enumerate a contiguous run-id range (optionally filtered
      by experiment name) and skip missing/aborted runs.
    * Incremental growth (``add_runs``) and a per-run parameter audit
      (``param_report``).

Limitations:
    * Only 1-D scans are supported. Every input must have ``Nvars == 1``.
    * All inputs must share the same ``xvarnames[0]`` unless
      ``xvarname_override=True``.
    * All inputs must share ``imaging_type`` and per-shot image shape.
      ``N_repeats`` may now differ across inputs (see ``merge_overlap``).
    * Shuffling, reassigning repeats, and transposing are disabled on the
      resulting vault.
"""

import copy
import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np

from waxa.atomdata import atomdata
from waxa.atomdata_base import (
    atomdata_base,
    analysis_tags,
)
from waxa.roi import ROI

if TYPE_CHECKING:
    # Used only for Pylance autocomplete on .avg / .std / .sem.
    _AvgType = atomdata_base


def _flatten_inputs(inputs):
    """Flatten a scalar / range / list / tuple / ndarray into a list."""
    if inputs is None:
        raise ValueError("AtomdataVault requires at least one input.")
    if isinstance(inputs, range):
        return np.asarray(inputs).ravel().tolist()
    if isinstance(inputs, (list, tuple)):
        out = []
        for item in inputs:
            out.extend(_flatten_inputs(item))
        return out
    if isinstance(inputs, np.ndarray):
        return [x for x in inputs.ravel().tolist()]
    return [inputs]


def _decode_xvarname(name):
    """Normalize an xvarname stored as bytes/np.bytes_/str to a plain str."""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    if isinstance(name, np.bytes_):
        return name.decode("utf-8", errors="replace")
    return str(name)


class _VaultDataVault():
    """Mimics the ``DataVault`` shape used by atomdata_base (a ``keys`` list
    plus arbitrary array attributes)."""

    def __init__(self):
        self.keys = []


class AtomdataVault(atomdata_base):
    """A virtual atomdata built by concatenating several single-axis runs.

    Parameters
    ----------
    inputs : atomdata, int, or (list/tuple/ndarray of) those
        Each entry is either an already-loaded ``atomdata`` object or a
        ``run_id`` (positive int). Run-ids are loaded internally using
        ``lite`` and ``roi_id``. Nested lists/tuples/ndarrays are flattened.
    roi_id : None, int, or str
        Forwarded to ``atomdata(...)`` when loading run-ids, and used for the
        vault's ROI. If ``None``, the ROI of the first input is used.
    lite : bool
        Forwarded to ``atomdata(...)`` when loading run-ids.
    xvarname_override : bool
        If True, skip the requirement that all inputs share the same
        ``xvarnames[0]`` and use the first input's name. A warning is emitted.
    sort : bool
        If True (default), the concatenated xvar axis is sorted ascending
        (stable) and every per-shot array is reindexed accordingly.
    merge_overlap : bool
        If True (default), ``vault.avg`` / ``vault.std`` / ``vault.sem`` group
        shots by unique xvar value rather than by a fixed repeat count. This is
        what makes overlapping ranges with ragged repeat counts average
        correctly. If False, the base (uniform-repeat) statistics are used and
        ragged counts fall back to a passthrough mean with zero spread.
    drop_raw_images : bool
        If True, free the (large) raw image stack after the initial analysis
        completes, keeping only derived quantities (``od``, ``atom_number``,
        fits, ...). Useful for many-run jobs. Defaults to False.
    auto_lite_threshold : int or None
        If set and more than this many run-ids are passed (and ``lite`` is
        False), automatically load them ``lite`` and emit a warning. Defaults
        to 8. Pass ``None`` to disable.
    ignore_images : bool
        If True, camera images are not loaded/concatenated and no ROI is
        created. Only the non-image data (params, DataVault fields,
        scope_data, xvars) is stitched together. Image-based analysis is
        skipped and image-derived attributes are set to ``None``.
    scope_merge : {'strict', 'pad_nan', 'skip'}
        How to concatenate ``scope_data`` traces. ``'strict'`` preserves the
        previous behavior and skips all scope data if trace dimensions differ.
        ``'pad_nan'`` pads shorter traces along the sample axis before
        concatenating shots. ``'skip'`` ignores scope data entirely.
    structure : {'prompt', 'auto', 'manual', None}
        How to handle scalar fixed parameters that differ across source runs.
        ``'auto'`` promotes the only clear disagreement to the first xvar axis
        without asking (default); ``'prompt'`` asks first; ``'manual'`` or
        ``None`` leaves the vault flat until ``set_xvar`` is called.
    xvar_mode : {'rectangular', 'pad'}
        Grid policy used by automatic/prompted ``set_xvar``. ``'rectangular'``
        requires every promoted-param value to share the same existing xvar
        sequence. ``'pad'`` preserves differing sequences in a padded grid and
        currently requires ``ignore_images=True``.
    promote_xvar : str or None
        Explicit scalar fixed-parameter key to promote to the first xvar axis
        after loading. If given, this takes precedence over ``structure``.
    flatten_xvar : str, int, or None
        Structured xvar key/index to flatten automatically after any promotion.
    """

    def __init__(self,
                 inputs,
                 roi_id=None,
                 lite=False,
                 xvarname_override=False,
                 sort=True,
                 merge_overlap=True,
                 drop_raw_images=False,
                 auto_lite_threshold=8,
                 ignore_images=False,
                 scope_merge='pad_nan',
                 structure='auto',
                 xvar_mode='pad',
                 promote_xvar=None,
                 flatten_xvar=None):

        # Lightweight book-keeping expected by inherited helpers.
        self._lite = lite
        self._ignore_images = bool(ignore_images)
        self._timing_enabled = False
        self._timing = {}
        self.server_talk = None

        # Vault-specific configuration.
        self._merge_overlap = bool(merge_overlap)
        self._drop_raw_images = bool(drop_raw_images)
        if scope_merge not in ('strict', 'pad_nan', 'skip'):
            raise ValueError(
                "scope_merge must be one of 'strict', 'pad_nan', or 'skip'."
            )
        self._scope_merge = scope_merge
        self._structure = structure
        self._xvar_mode = xvar_mode
        # Kwargs needed to rebuild an equivalent vault (used by add_runs).
        self._build_kwargs = dict(
            roi_id=roi_id,
            lite=lite,
            xvarname_override=xvarname_override,
            sort=sort,
            merge_overlap=merge_overlap,
            drop_raw_images=drop_raw_images,
            auto_lite_threshold=auto_lite_threshold,
            ignore_images=ignore_images,
            scope_merge=scope_merge,
            structure=structure,
            xvar_mode=xvar_mode,
            promote_xvar=promote_xvar,
            flatten_xvar=flatten_xvar,
        )

        self.avg: Optional[atomdata_base] = None
        self.std: Optional[atomdata_base] = None
        self.sem: Optional[atomdata_base] = None
        self._repeat_sem_source = None
        self._repeat_sem_divisor = None
        self._repeat_zero_proxy = None
        self._repeat_lazy_stat_context = None
        self._data_file_path = None
        self._saved_roi_from_file = False
        # Per-run parameter audit, filled in by _warn_param_mismatches.
        self.param_disagreements = {}
        self.source_param_values = {}
        self._shot_param_values = {}
        self._structured_xvars = False

        # 1. Normalize and materialize inputs.
        raw_inputs = _flatten_inputs(inputs)
        if len(raw_inputs) == 0:
            raise ValueError("AtomdataVault requires at least one input.")

        # Memory guard: when many run-ids are requested, default to loading the
        # pre-cropped lite datasets unless the caller explicitly opted out.
        n_int_inputs = sum(
            isinstance(item, (int, np.integer)) for item in raw_inputs
        )
        if (auto_lite_threshold is not None
                and not lite
                and n_int_inputs > int(auto_lite_threshold)):
            warnings.warn(
                f"AtomdataVault: loading {n_int_inputs} run-ids; switching to "
                f"lite datasets to limit memory (auto_lite_threshold="
                f"{auto_lite_threshold}). Pass auto_lite_threshold=None to "
                f"disable, or lite=True to silence.",
                stacklevel=2,
            )
            lite = True
            self._lite = True
            self._build_kwargs['lite'] = True

        # Load run-ids sequentially so that only the first load can trigger
        # the ROI selection GUI. After the first run is loaded (or if the
        # first input is an already-loaded atomdata), its run_id is used as
        # the roi_id for all subsequent int loads, ensuring one consistent ROI
        # is reused rather than opening a new selector for each chunk.
        ads = []
        _first_run_id = None  # run_id of the first materialized atomdata
        _has_subsequent_int_loads = any(
            isinstance(item, (int, np.integer)) for item in raw_inputs[1:]
        )

        for item in raw_inputs:
            if isinstance(item, atomdata_base):
                ads.append(item)
                if _first_run_id is None:
                    _first_run_id = int(item.run_info.run_id)
                    # Save the ROI so subsequent int loads (and lite-dataset
                    # creation) can look it up by run_id.
                    if (not self._ignore_images) and (_has_subsequent_int_loads or lite):
                        item.save_roi_h5()
            elif isinstance(item, (int, np.integer)):
                # First int load: use caller-supplied roi_id (may open GUI once).
                # Subsequent int loads: reuse the first run's roi_id so no
                # additional GUI opens.
                if _first_run_id is None:
                    ad = atomdata(int(item), roi_id=roi_id, lite=lite,
                                  ignore_images=self._ignore_images)
                    _first_run_id = int(ad.run_info.run_id)
                    # Persist the ROI so subsequent int loads (and lite-dataset
                    # creation for each run) can find it by run_id.
                    if (not self._ignore_images) and (_has_subsequent_int_loads or lite):
                        ad.save_roi_h5()
                else:
                    _roi = roi_id if roi_id is not None else _first_run_id
                    ad = atomdata(int(item), roi_id=_roi, lite=lite,
                                  ignore_images=self._ignore_images)
                ads.append(ad)
            else:
                raise TypeError(
                    f"AtomdataVault inputs must be atomdata objects or "
                    f"run_id ints, got {type(item).__name__}."
                )

        # 2. Validate compatibility. N_repeats may differ across inputs when
        #    merge_overlap is on (grouped statistics handle ragged counts).
        self._validate_inputs(
            ads, xvarname_override,
            allow_repeat_mismatch=self._merge_overlap,
            ignore_images=self._ignore_images,
        )

        # 3. Unshuffle each chunk so the per-shot arrays are in xvar order on
        #    axis 0. Only chunks that actually need unshuffling are deep-copied
        #    (unshuffle mutates in place); already-ordered chunks are used
        #    by-reference and only read from, which avoids duplicating large
        #    image stacks for the common many-run / lite case.
        chunks = []
        for ad in ads:
            if getattr(ad._analysis_tags, 'xvars_shuffled', False):
                ad_copy = copy.deepcopy(ad)
                ad_copy.unshuffle(reanalyze=False)
                if getattr(ad_copy, '_has_images', True):
                    ad_copy._sort_images()
                chunks.append(ad_copy)
            else:
                chunks.append(ad)

        # 4. Assemble vault state from chunks.
        first = chunks[0]
        xvarname = _decode_xvarname(first.xvarnames[0])
        self.source_run_ids = [int(c.run_info.run_id) for c in chunks]
        self._source_atomdata_by_run_id = {
            int(c.run_info.run_id): c for c in chunks
        }

        # params: deep-copy first, then patch the scanned attribute below.
        self.params = copy.deepcopy(first.params)
        self.p = self.params
        self.camera_params = copy.deepcopy(first.camera_params)
        self.run_info = copy.deepcopy(first.run_info)
        self.run_info.run_id = list(self.source_run_ids)
        self.experiment_code = getattr(first, 'experiment_code', None)
        self._has_images = (
            False if self._ignore_images
            else bool(getattr(first, '_has_images', True))
        )

        self._warn_param_mismatches(chunks)

        # Concatenate xvar values (axis 0).
        xvar_values = np.concatenate(
            [np.asarray(c.xvars[0]) for c in chunks], axis=0
        )

        # Per-shot provenance: which source run each concatenated shot came
        # from. Carried through the sort/reindex below so it always lines up
        # with the analyzed arrays.
        self.shot_run_id = np.concatenate([
            np.full(int(np.asarray(c.xvars[0]).shape[0]),
                    int(c.run_info.run_id), dtype=np.int64)
            for c in chunks
        ])
        self._build_shot_param_values(chunks)

        # Concatenate images / timestamps if present.
        # Keep a reference to the first chunk's raw images so the ROI GUI
        # shows a representative frame from the first run only (not all runs).
        if self._has_images:
            _first_chunk_images = np.asarray(chunks[0].images)
            self.images = np.concatenate(
                [np.asarray(c.images) for c in chunks], axis=0
            )
            self.image_timestamps = np.concatenate(
                [np.asarray(c.image_timestamps) for c in chunks], axis=0
            )
        else:
            _first_chunk_images = None
            self.images = np.array([])
            self.image_timestamps = np.array([])

        # Concatenate DataVault containers (union across chunks; NaN-pad
        # missing entries).
        self.data = self._concat_data_vaults(chunks)

        # Concatenate scope_data only if every chunk has it (and contains the
        # same scope/channel keys). Otherwise emit a warning and skip.
        self._maybe_concat_scope_data(chunks, mode=self._scope_merge)

        # Patch the scanned param to the concatenated array.
        setattr(self.params, xvarname, xvar_values)

        # Update shot-count params so the dealer reshapes the concatenated
        # images correctly. For a 1-D scan, len(xvar_values) is the total
        # per-shot count including repeats.
        total_shots = int(len(xvar_values))
        self.params.N_shots_with_repeats = total_shots
        if hasattr(self.params, 'N_shots'):
            nrep = int(getattr(self.params, 'N_repeats', 1) or 1)
            self.params.N_shots = total_shots // nrep if nrep > 0 else total_shots

        # xvar scaffolding for the (single) scan axis.
        self.xvarnames = [xvarname]
        self.xvars = [xvar_values]
        self.xvardims = np.array([total_shots], dtype=int)
        self.Nvars = 1

        # Vault is permanently unshuffled.
        self.sort_idx = np.array([])
        self.sort_N = np.array([])

        # 5. Optional sort along the merged axis.
        if sort:
            self._sort_axis0_by_xvar()

        # 6. Build helper objects expected by _initial_analysis.
        from waxa.data.data_saver import DataSaver
        self._ds = DataSaver()
        self._dealer = self._init_dealer()
        self._analysis_tags = analysis_tags(
            roi_id=roi_id,
            imaging_type=self.run_info.imaging_type,
        )
        self._analysis_tags.xvars_shuffled = False

        # ROI: reuse the first chunk's already-resolved ROI so all runs are
        # cropped with exactly the ROI chosen for the first run (loaded from
        # its h5 file, or selected via the GUI at load time). This guarantees
        # a single, consistent ROI across every run without re-opening the
        # selection GUI. The resulting roix/roiy coordinates are applied to
        # the full concatenated od_raw during analyze_ods.
        if self._has_images:
            first_roi = getattr(first, 'roi', None)
            if first_roi is not None:
                self.roi = copy.deepcopy(first_roi)
                # Point the ROI at the first chunk's frame for any later GUI
                # display (e.g. recrop) and at the first run's id.
                self.roi._images = _first_chunk_images
                self.roi.run_id = int(first.run_info.run_id)
                self.roi._current_file_path = None
                self.roi._current_saved_roi = [self.roi.roix, self.roi.roiy]
            else:
                roi_source = (
                    roi_id if roi_id is not None
                    else int(first.run_info.run_id)
                )
                self.roi = ROI(
                    run_id=int(first.run_info.run_id),
                    roi_id=roi_source,
                    use_saved_roi=True,
                    lite=self._lite,
                    server_talk=None,
                    current_file_path=None,
                    current_saved_roi=None,
                    images=_first_chunk_images,
                    imaging_type=self.run_info.imaging_type,
                )
        else:
            self.roi = None

        # 7. Run the standard initial analysis pipeline.
        self._initial_analysis(transpose_idx=[], avg_repeats=False)

        if promote_xvar is not None:
            self.set_xvar(
                promote_xvar,
                xvar_mode=xvar_mode,
                refresh_statistics=flatten_xvar is None,
            )
        else:
            self._maybe_structure_from_param_disagreements(structure, xvar_mode)

        if flatten_xvar is not None:
            self.flatten_xvar(flatten_xvar)

        # 8. Optionally free the raw image stack now that derived quantities
        #    (od, atom_number, fits, ...) have been computed.
        if self._drop_raw_images and self._has_images:
            self.images = np.array([])
            self.image_timestamps = np.array([])

    def _initial_analysis(self, transpose_idx, avg_repeats):
        """Mirror ``atomdata._initial_analysis``: skip all image-based analysis
        for runs that captured no camera images (e.g. APD/scope-only runs),
        otherwise defer to the standard base pipeline."""
        if not getattr(self, '_has_images', True):
            self._clear_image_analysis_attrs()
            self._refresh_repeat_statistics()
            return
        return atomdata_base._initial_analysis(self, transpose_idx, avg_repeats)

    def _maybe_structure_from_param_disagreements(self, structure, xvar_mode):
        if structure in (None, False, 'manual'):
            return
        if structure not in ('prompt', 'auto'):
            raise ValueError(
                "structure must be one of 'prompt', 'auto', 'manual', None, "
                f"or False; got {structure!r}."
            )

        candidates = sorted(getattr(self, '_shot_param_values', {}).keys())
        if len(candidates) == 0:
            return
        if len(candidates) > 1:
            warnings.warn(
                "AtomdataVault: multiple scalar fixed parameters differ across "
                "input runs; leaving data flat. Choose one with "
                "vault.set_xvar(param_key). Available keys: "
                + ", ".join(candidates),
                stacklevel=2,
            )
            return

        param_key = candidates[0]
        if structure == 'prompt':
            try:
                answer = input(
                    "AtomdataVault: promote differing parameter "
                    f"{param_key!r} to the first xvar axis? [y/N] "
                )
            except EOFError:
                warnings.warn(
                    "AtomdataVault: could not prompt for xvar structure; "
                    f"leaving data flat. Call vault.set_xvar({param_key!r}) "
                    "to structure it manually.",
                    stacklevel=2,
                )
                return
            if answer.strip().lower() not in ('y', 'yes'):
                return

        self.set_xvar(param_key, xvar_mode=xvar_mode)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, ads, xvarname_override, allow_repeat_mismatch=False, ignore_images=False):
        first = ads[0]

        if int(getattr(first, 'Nvars', 0)) != 1:
            raise ValueError(
                "AtomdataVault only supports single-axis (1-D) scans; "
                f"first input has Nvars={first.Nvars}."
            )

        first_name = _decode_xvarname(first.xvarnames[0])
        first_nrep = int(first.params.N_repeats)
        first_imgtype = first.run_info.imaging_type
        first_has_images = bool(getattr(first, '_has_images', True))
        first_img_shape = (
            tuple(np.asarray(first.images).shape[1:])
            if (first_has_images and not ignore_images) else None
        )

        repeat_counts = {int(first.run_info.run_id): first_nrep}

        for ad in ads[1:]:
            if int(getattr(ad, 'Nvars', 0)) != 1:
                raise ValueError(
                    "AtomdataVault only supports single-axis (1-D) scans; "
                    f"run {ad.run_info.run_id} has Nvars={ad.Nvars}."
                )
            name = _decode_xvarname(ad.xvarnames[0])
            if name != first_name and not xvarname_override:
                raise ValueError(
                    f"xvarname mismatch: run {ad.run_info.run_id} has "
                    f"'{name}' but first input has '{first_name}'. "
                    f"Pass xvarname_override=True to override."
                )
            ad_nrep = int(ad.params.N_repeats)
            repeat_counts[int(ad.run_info.run_id)] = ad_nrep
            if ad_nrep != first_nrep and not allow_repeat_mismatch:
                raise ValueError(
                    f"N_repeats mismatch: run {ad.run_info.run_id} has "
                    f"N_repeats={ad.params.N_repeats} but first input has "
                    f"N_repeats={first_nrep}. Pass merge_overlap=True (the "
                    f"default) to allow ragged repeat counts."
                )
            if ad.run_info.imaging_type != first_imgtype:
                raise ValueError(
                    f"imaging_type mismatch on run {ad.run_info.run_id}."
                )
            # Image presence/shape checks are irrelevant when images are
            # ignored entirely.
            if ignore_images:
                continue
            ad_has_images = bool(getattr(ad, '_has_images', True))
            if ad_has_images != first_has_images:
                raise ValueError(
                    f"Image-presence mismatch on run {ad.run_info.run_id}."
                )
            if first_has_images:
                ad_shape = tuple(np.asarray(ad.images).shape[1:])
                if ad_shape != first_img_shape:
                    raise ValueError(
                        f"Image shape mismatch: run {ad.run_info.run_id} "
                        f"has per-shot shape {ad_shape} but first input has "
                        f"{first_img_shape}."
                    )

        if xvarname_override:
            names = [_decode_xvarname(a.xvarnames[0]) for a in ads]
            unique = sorted(set(names))
            if len(unique) > 1:
                warnings.warn(
                    f"xvarname_override=True: using '{first_name}' but inputs "
                    f"had xvarnames {unique}.",
                    stacklevel=2,
                )

        # Record per-run repeat counts for the parameter audit.
        self.source_repeat_counts = repeat_counts

    def _warn_param_mismatches(self, chunks):
        """Emit a single warning summarizing fixed-param disagreements
        across chunks (excluding the scanned xvar itself) and record the
        per-run values in ``self.param_disagreements`` for ``param_report``."""
        first = chunks[0]
        first_params = vars(first.params)
        xvarname = _decode_xvarname(first.xvarnames[0])

        def _equalish(a_val, b_val):
            try:
                if isinstance(a_val, np.ndarray) or isinstance(b_val, np.ndarray):
                    a = np.asarray(a_val)
                    b = np.asarray(b_val)
                    return a.shape == b.shape and np.array_equal(a, b)
                return a_val == b_val
            except Exception:
                # Non-comparable params are treated as "equal" (skipped).
                return True

        mismatched = []
        disagreements = {}
        for key, first_val in first_params.items():
            if key.startswith('_') or key == xvarname:
                continue
            differs = False
            for c in chunks[1:]:
                other = vars(c.params).get(key, None)
                if other is None:
                    continue
                if not _equalish(first_val, other):
                    differs = True
                    break
            if differs:
                mismatched.append(key)
                disagreements[key] = {
                    int(c.run_info.run_id): vars(c.params).get(key, None)
                    for c in chunks
                }

        self.param_disagreements = disagreements
        self.source_param_values = disagreements

        if mismatched:
            warnings.warn(
                "AtomdataVault: fixed parameters disagree across input runs "
                "(using values from the first run): "
                + ", ".join(sorted(set(mismatched)))
                + ". Call vault.param_report() for a per-run breakdown.",
                stacklevel=2,
            )

    def _build_shot_param_values(self, chunks):
        """Broadcast scalar per-run disagreement values onto the shot axis."""
        shot_values = {}
        for key in self.param_disagreements:
            pieces = []
            scalar_values = True
            for c in chunks:
                value = vars(c.params).get(key, None)
                arr = np.asarray(value)
                if arr.ndim != 0:
                    scalar_values = False
                    break
                n_shots = int(np.asarray(c.xvars[0]).shape[0])
                pieces.append(np.full(n_shots, arr.item()))
            if scalar_values and pieces:
                shot_values[key] = np.concatenate(pieces, axis=0)
        self._shot_param_values = shot_values

    # ------------------------------------------------------------------
    # Concatenation helpers
    # ------------------------------------------------------------------
    def _concat_data_vaults(self, chunks):
        """Union of every chunk's data.keys, NaN-padding missing chunks."""
        dv = _VaultDataVault()

        # Collect all keys in first-seen order.
        ordered_keys = []
        seen = set()
        for c in chunks:
            for k in c.data.keys:
                if k not in seen:
                    ordered_keys.append(k)
                    seen.add(k)

        # Per-chunk axis-0 length for padding.
        chunk_lengths = [int(np.asarray(c.xvars[0]).shape[0]) for c in chunks]

        for k in ordered_keys:
            pieces = []
            template = None
            template_chunk = None
            for c, n in zip(chunks, chunk_lengths):
                if k in c.data.keys:
                    arr = np.asarray(vars(c.data)[k])
                    pieces.append(('arr', arr))
                    if template is None:
                        template = arr
                        template_chunk = n
                else:
                    pieces.append(('pad', n))

            if template is None:
                # No chunk actually has the key — skip.
                continue

            trailing_shape = template.shape[1:] if template.ndim >= 1 else ()
            # NaN padding requires a float dtype.
            if pieces and any(p[0] == 'pad' for p in pieces):
                if np.issubdtype(template.dtype, np.floating):
                    dtype = template.dtype
                else:
                    dtype = np.float64
            else:
                dtype = template.dtype

            built = []
            for tag, payload in pieces:
                if tag == 'arr':
                    arr = payload
                    if arr.dtype != dtype:
                        arr = arr.astype(dtype, copy=False)
                    built.append(arr)
                else:
                    n = payload
                    pad_shape = (n,) + tuple(trailing_shape)
                    pad = np.full(pad_shape, np.nan, dtype=dtype)
                    built.append(pad)

            try:
                concatenated = np.concatenate(built, axis=0)
            except ValueError as e:
                warnings.warn(
                    f"AtomdataVault: skipping data key '{k}' because "
                    f"chunks could not be concatenated: {e}.",
                    stacklevel=2,
                )
                continue

            vars(dv)[k] = concatenated
            dv.keys.append(k)

        return dv

    @staticmethod
    def _pad_scope_array(arr, target_len):
        arr = np.asarray(arr)
        if arr.ndim < 2:
            raise ValueError(
                f"expected a scope trace shaped (n_shots, n_samples), got {arr.shape}"
            )
        arr = arr.astype(np.float64, copy=False)
        if arr.shape[-1] == target_len:
            return arr
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, int(target_len) - int(arr.shape[-1]))
        return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)

    @staticmethod
    def _scope_arrays_pad_compatible(parts):
        shapes = [np.asarray(p).shape for p in parts]
        if any(len(s) < 2 for s in shapes):
            return False
        base_ndim = len(shapes[0])
        base_middle = shapes[0][1:-1]
        return all(len(s) == base_ndim and s[1:-1] == base_middle for s in shapes)

    def _maybe_concat_scope_data(self, chunks, mode='strict'):
        if mode == 'skip':
            return
        if not all(hasattr(c, 'scope_data') and bool(c.scope_data) for c in chunks):
            return

        first_scope = chunks[0].scope_data
        scope_keys = list(first_scope.keys())
        for c in chunks[1:]:
            if list(c.scope_data.keys()) != scope_keys:
                warnings.warn(
                    "AtomdataVault: scope_data keys differ across chunks; "
                    "skipping scope_data concatenation.",
                    stacklevel=2,
                )
                return

        # Concat per scope_key / per channel for 't' and 'v'.
        from waxa.atomdata_base import ScopeTraceArray
        merged = {}
        try:
            for scope_key in scope_keys:
                first_channels = first_scope[scope_key]
                ch_keys = list(first_channels.keys())
                merged[scope_key] = {}
                for ch in ch_keys:
                    t_parts = [np.asarray(c.scope_data[scope_key][ch].t) for c in chunks]
                    v_parts = [np.asarray(c.scope_data[scope_key][ch].v) for c in chunks]
                    if mode == 'pad_nan':
                        if (not self._scope_arrays_pad_compatible(t_parts)
                                or not self._scope_arrays_pad_compatible(v_parts)):
                            warnings.warn(
                                f"AtomdataVault: scope_data arrays for scope "
                                f"'{scope_key}' channel {ch} have incompatible "
                                f"shapes; skipping scope_data concatenation.",
                                stacklevel=2,
                            )
                            return
                        t_lengths = [int(t.shape[-1]) for t in t_parts]
                        v_lengths = [int(v.shape[-1]) for v in v_parts]
                        target_len = max(max(t_lengths), max(v_lengths))
                        if len(set(t_lengths + v_lengths)) > 1:
                            warnings.warn(
                                f"AtomdataVault: scope_data traces for scope "
                                f"'{scope_key}' channel {ch} have different "
                                f"sample lengths {sorted(set(t_lengths + v_lengths))}; "
                                f"padding shorter traces with NaN.",
                                stacklevel=2,
                            )
                        t_parts = [self._pad_scope_array(t, target_len) for t in t_parts]
                        v_parts = [self._pad_scope_array(v, target_len) for v in v_parts]
                    t_cat = np.concatenate(t_parts, axis=0)
                    v_cat = np.concatenate(v_parts, axis=0)
                    merged[scope_key][ch] = ScopeTraceArray(scope_key, ch, t_cat, v_cat)
        except Exception as e:
            warnings.warn(
                f"AtomdataVault: failed to concatenate scope_data ({e}); "
                "skipping.",
                stacklevel=2,
            )
            return

        self.scope_data = merged

    def _sort_axis0_by_xvar(self):
        """Sort every per-shot array on axis 0 by ascending xvar value."""
        xvar_values = np.asarray(self.xvars[0])
        order = np.argsort(xvar_values, kind='stable')
        if np.array_equal(order, np.arange(len(xvar_values))):
            return  # already sorted
        self._reorder_shots(order)

    def _reorder_shots(self, order):
        """Reindex every per-shot quantity along axis 0 by ``order``.

        ``order`` is an array of source shot-indices to keep (in the desired
        new order). Used both for the construction-time sort (a permutation)
        and for ``drop_runs`` (a subset). Handles the merged xvar, per-shot
        provenance, raw interleaved or 1-per-shot images, DataVault arrays,
        scope traces, and any top-level scan-shaped analysis arrays
        (``od_raw``, ``atom_number``, ...) that already exist.
        """
        order = np.asarray(order, dtype=int)
        n_old = int(np.asarray(self.xvars[0]).shape[0])
        n_new = int(order.shape[0])

        # Merged xvar + scanned param.
        self.xvars[0] = np.asarray(self.xvars[0])[order]
        setattr(self.params, self.xvarnames[0], self.xvars[0])

        # Per-shot provenance.
        if hasattr(self, 'shot_run_id'):
            self.shot_run_id = np.asarray(self.shot_run_id)[order]

        for key, values in list(getattr(self, '_shot_param_values', {}).items()):
            values = np.asarray(values)
            if values.ndim >= 1 and values.shape[0] == n_old:
                self._shot_param_values[key] = values[order]

        # Images / timestamps (raw interleaved or 1-per-shot).
        if self._has_images and np.asarray(self.images).size:
            Nf = int(self.params.N_pwa_per_shot) + 2
            if self.images.shape[0] == n_old * Nf:
                if n_new:
                    frame_order = np.concatenate(
                        [np.arange(i * Nf, (i + 1) * Nf) for i in order]
                    )
                else:
                    frame_order = np.array([], dtype=int)
                self.images = self.images[frame_order]
                self.image_timestamps = self.image_timestamps[frame_order]
            elif self.images.shape[0] == n_old:
                self.images = self.images[order]
                self.image_timestamps = self.image_timestamps[order]

        # DataVault arrays.
        for k in self.data.keys:
            arr = vars(self.data)[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_old:
                vars(self.data)[k] = arr[order]

        # Scope traces.
        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    t = np.asarray(trace.t)
                    v = np.asarray(trace.v)
                    if t.ndim >= 1 and t.shape[0] == n_old:
                        trace.t = t[order]
                    if v.ndim >= 1 and v.shape[0] == n_old:
                        trace.v = v[order]

        # Top-level scan-shaped analysis arrays (present post-analysis).
        _skip = {'images', 'image_timestamps', 'shot_run_id'}
        for key, val in list(vars(self).items()):
            if key in _skip or key.startswith('_'):
                continue
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_old:
                vars(self)[key] = val[order]

        # Update shot-count bookkeeping (matters when n_new != n_old).
        self.xvardims = np.array([n_new], dtype=int)
        nrep = int(getattr(self.params, 'N_repeats', 1) or 1)
        self.params.N_shots_with_repeats = n_new
        if hasattr(self.params, 'N_shots'):
            self.params.N_shots = n_new // nrep if nrep > 0 else n_new

    def _reshape_axis0_to_first_xvar(self, arr, order, new_shape):
        arr = np.asarray(arr)
        if arr.ndim < 1 or arr.shape[0] != len(order):
            return arr
        arr = arr[np.asarray(order, dtype=int)]
        return arr.reshape(*new_shape, *arr.shape[1:])

    def _reshape_structured_axis0_arrays(self, order, new_shape):
        n_old = len(order)
        self.shot_run_id = self._reshape_axis0_to_first_xvar(
            self.shot_run_id, order, new_shape
        )

        for key, values in list(getattr(self, '_shot_param_values', {}).items()):
            values = np.asarray(values)
            if values.ndim >= 1 and values.shape[0] == n_old:
                self._shot_param_values[key] = self._reshape_axis0_to_first_xvar(
                    values, order, new_shape
                )

        if self._has_images and np.asarray(self.images).size:
            # Raw camera images live in a flat (N_img, H, W) buffer that the
            # dealer re-structures from xvardims, so they must be *reordered*
            # to the new shot raster order but kept flat -- not reshaped onto
            # the structured leading axes (which would break deal_data_ndarray).
            Nf = int(self.params.N_pwa_per_shot) + 2
            if self.images.shape[0] == n_old * Nf:
                # Interleaved raw frames: expand each shot index to its Nf frames.
                if n_old:
                    frame_order = np.concatenate(
                        [np.arange(i * Nf, (i + 1) * Nf)
                         for i in np.asarray(order, dtype=int)]
                    )
                else:
                    frame_order = np.array([], dtype=int)
                self.images = self.images[frame_order]
                self.image_timestamps = self.image_timestamps[frame_order]
            elif self.images.shape[0] == n_old:
                # One (already-processed) image per shot: promote onto the
                # structured leading axes like the other scan-shaped arrays.
                self.images = self._reshape_axis0_to_first_xvar(
                    self.images, order, new_shape
                )
                self.image_timestamps = self._reshape_axis0_to_first_xvar(
                    self.image_timestamps, order, new_shape
                )

        for key in self.data.keys:
            arr = vars(self.data)[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_old:
                vars(self.data)[key] = self._reshape_axis0_to_first_xvar(
                    arr, order, new_shape
                )

        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    t = np.asarray(trace.t)
                    v = np.asarray(trace.v)
                    if t.ndim >= 1 and t.shape[0] == n_old:
                        trace.t = self._reshape_axis0_to_first_xvar(
                            t, order, new_shape
                        )
                    if v.ndim >= 1 and v.shape[0] == n_old:
                        trace.v = self._reshape_axis0_to_first_xvar(
                            v, order, new_shape
                        )

        _skip = {
            'images', 'image_timestamps', 'shot_run_id', 'xvars', 'xvarnames',
            'xvardims', 'data', 'scope_data', 'params', 'p', 'camera_params',
            'run_info', 'roi', 'avg', 'std', 'sem', 'sort_idx', 'sort_N',
        }
        for key, val in list(vars(self).items()):
            if key in _skip or key.startswith('_'):
                continue
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_old:
                vars(self)[key] = self._reshape_axis0_to_first_xvar(
                    val, order, new_shape
                )

    @staticmethod
    def _build_padded_order_grid(param_values, old_xvar, param_unique, sort=True):
        row_orders = []
        max_len = 0
        for param_value in param_unique:
            idx = np.where(param_values == param_value)[0]
            if idx.size == 0:
                raise ValueError(f'No shots found for promoted xvar value {param_value!r}.')
            if sort:
                idx = idx[np.argsort(old_xvar[idx], kind='stable')]
            row_orders.append(idx)
            max_len = max(max_len, int(idx.size))

        order_grid = np.full((len(param_unique), max_len), -1, dtype=int)
        old_xvar_grid = np.full((len(param_unique), max_len), np.nan, dtype=np.float64)
        for row_idx, idx in enumerate(row_orders):
            n = int(idx.size)
            order_grid[row_idx, :n] = idx
            old_xvar_grid[row_idx, :n] = np.asarray(old_xvar, dtype=np.float64)[idx]
        return order_grid, old_xvar_grid

    @staticmethod
    def _pad_axis0_ndarray(arr, order_grid):
        arr = np.asarray(arr)
        if arr.ndim < 1:
            return arr
        valid = order_grid >= 0
        trailing_shape = arr.shape[1:]
        out_shape = tuple(order_grid.shape) + trailing_shape
        if np.issubdtype(arr.dtype, np.number):
            out = np.full(out_shape, np.nan, dtype=np.float64)
        else:
            out = np.full(out_shape, None, dtype=object)
        if np.any(valid):
            out[valid] = arr[order_grid[valid]]
        return out

    @staticmethod
    def _pad_scope_trace_axis0(arr, order_grid):
        arr = np.asarray(arr)
        if arr.ndim < 1:
            return arr
        out = np.full(tuple(order_grid.shape), None, dtype=object)
        valid = order_grid >= 0
        if np.any(valid):
            rows = arr[order_grid[valid]]
            for idx, row in zip(zip(*np.where(valid)), rows):
                out[idx] = row
        return out

    def _reshape_padded_axis0_arrays(self, order_grid):
        n_old = int(np.max(order_grid)) + 1 if np.any(order_grid >= 0) else 0

        if hasattr(self, 'shot_run_id'):
            self.shot_run_id = self._pad_axis0_ndarray(
                np.asarray(self.shot_run_id, dtype=object), order_grid
            )

        for key, values in list(getattr(self, '_shot_param_values', {}).items()):
            values = np.asarray(values)
            if values.ndim >= 1 and values.shape[0] == n_old:
                self._shot_param_values[key] = self._pad_axis0_ndarray(
                    values, order_grid
                )

        for key in self.data.keys:
            arr = vars(self.data)[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_old:
                vars(self.data)[key] = self._pad_axis0_ndarray(arr, order_grid)

        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    t = np.asarray(trace.t)
                    v = np.asarray(trace.v)
                    if t.ndim >= 1 and t.shape[0] == n_old:
                        trace.t = self._pad_scope_trace_axis0(t, order_grid)
                    if v.ndim >= 1 and v.shape[0] == n_old:
                        trace.v = self._pad_scope_trace_axis0(v, order_grid)

        _skip = {
            'images', 'image_timestamps', 'shot_run_id', 'xvars', 'xvarnames',
            'xvardims', 'data', 'scope_data', 'params', 'p', 'camera_params',
            'run_info', 'roi', 'avg', 'std', 'sem', 'sort_idx', 'sort_N',
        }
        for key, val in list(vars(self).items()):
            if key in _skip or key.startswith('_'):
                continue
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_old:
                vars(self)[key] = self._pad_axis0_ndarray(val, order_grid)

    @staticmethod
    def _stable_unique(values):
        values = np.asarray(values)
        _, idx = np.unique(values, return_index=True)
        return values[np.sort(idx)]

    def set_xvar(self, param_key, xvar_idx=None, *, xvar_mode='rectangular',
                 sort=True, reanalyze=True, refresh_statistics=True):
        """Promote a scalar per-run parameter to the first xvar axis.

        ``xvar_mode='rectangular'`` supports the common builder pattern where
        each value of ``param_key`` owns the same ordered sequence of shots on
        the existing xvar. ``xvar_mode='pad'`` preserves uneven sequences in a
        padded two-axis grid and currently requires ``ignore_images=True``.
        """
        if xvar_idx not in (None, 0):
            raise NotImplementedError(
                'AtomdataVault.set_xvar currently always inserts the new xvar '
                'at axis 0; pass xvar_idx=None or xvar_idx=0.'
            )
        if xvar_mode not in ('rectangular', 'pad'):
            raise ValueError(
                "xvar_mode must be one of 'rectangular' or 'pad'; "
                f"got {xvar_mode!r}."
            )
        if int(getattr(self, 'Nvars', 0)) != 1:
            raise NotImplementedError(
                'AtomdataVault.set_xvar currently supports vaults with one '
                'existing xvar. Additional set_xvar calls are not implemented yet.'
            )
        if param_key not in getattr(self, '_shot_param_values', {}):
            available = sorted(getattr(self, '_shot_param_values', {}).keys())
            raise KeyError(
                f"No scalar per-shot values are available for param "
                f"{param_key!r}. Available keys: {available}."
            )

        param_values = np.asarray(self._shot_param_values[param_key])
        old_xvar = np.asarray(self.xvars[0])
        if param_values.shape[0] != old_xvar.shape[0]:
            raise ValueError(
                f"Param {param_key!r} has {param_values.shape[0]} shot values, "
                f"but the vault xvar has {old_xvar.shape[0]} shots."
            )
        if xvar_mode == 'pad' and getattr(self, '_has_images', True):
            raise NotImplementedError(
                "AtomdataVault.set_xvar(..., xvar_mode='pad') currently "
                "requires ignore_images=True. Rebuild the vault with "
                "ignore_images=True or use xvar_mode='rectangular'."
            )

        param_unique = self._stable_unique(param_values)
        if sort:
            try:
                param_unique = np.sort(param_unique)
            except TypeError:
                pass

        old_xvarname = self.xvarnames[0]
        if xvar_mode == 'pad':
            order_grid, old_xvar_grid = self._build_padded_order_grid(
                param_values, old_xvar, param_unique, sort=sort
            )
            self._reshape_padded_axis0_arrays(order_grid)

            self.xvarnames = [str(param_key), old_xvarname]
            self.xvars = [np.array(param_unique, copy=True), old_xvar_grid]
            self.xvardims = np.array(order_grid.shape, dtype=int)
            self.Nvars = 2
            setattr(self.params, str(param_key), self.xvars[0])
            setattr(self.params, old_xvarname, self.xvars[1])

            self.params.N_repeats = np.array([1, 1], dtype=int)
            self.params.N_shots_with_repeats = int(np.count_nonzero(order_grid >= 0))
            if hasattr(self.params, 'N_shots'):
                self.params.N_shots = int(np.count_nonzero(order_grid >= 0))

            self.sort_idx = np.array([])
            self.sort_N = np.array([])
            self._structured_xvars = True
            self._padded_xvar_mask = order_grid >= 0
            self._dealer = self._init_dealer()

            if not getattr(self, '_has_images', True):
                self._clear_image_analysis_attrs()
            if refresh_statistics:
                self._refresh_repeat_statistics()
            return self

        order_parts = []
        reference_xvar = None
        for param_value in param_unique:
            idx = np.where(param_values == param_value)[0]
            if idx.size == 0:
                raise ValueError(f'No shots found for {param_key}={param_value!r}.')
            if sort:
                idx = idx[np.argsort(old_xvar[idx], kind='stable')]
            this_xvar = old_xvar[idx]
            if reference_xvar is None:
                reference_xvar = np.array(this_xvar, copy=True)
            elif (this_xvar.shape != reference_xvar.shape
                    or not np.array_equal(this_xvar, reference_xvar)):
                raise ValueError(
                    f"Cannot promote {param_key!r} to an xvar because its "
                    "values do not form a rectangular grid with the existing "
                    f"xvar {self.xvarnames[0]!r}. Pass xvar_mode='pad' "
                    "or choose a different param."
                )
            order_parts.append(idx)

        order = np.concatenate(order_parts)
        new_shape = (len(param_unique), len(reference_xvar))
        self._reshape_structured_axis0_arrays(order, new_shape)

        self.xvarnames = [str(param_key), old_xvarname]
        self.xvars = [np.array(param_unique, copy=True), reference_xvar]
        self.xvardims = np.array(new_shape, dtype=int)
        self.Nvars = 2
        setattr(self.params, str(param_key), self.xvars[0])
        setattr(self.params, old_xvarname, self.xvars[1])

        _, counts = np.unique(reference_xvar, return_counts=True)
        repeated_counts = counts[counts > 1]
        n_repeats = 1
        if repeated_counts.size:
            unique_counts = np.unique(repeated_counts)
            if unique_counts.size == 1:
                n_repeats = int(unique_counts[0])
        self.params.N_repeats = np.array([1, n_repeats], dtype=int)
        self.params.N_shots_with_repeats = int(np.prod(self.xvardims))
        if hasattr(self.params, 'N_shots'):
            self.params.N_shots = int(len(param_unique) * len(np.unique(reference_xvar)))

        self.sort_idx = np.array([])
        self.sort_N = np.array([])
        self._structured_xvars = True
        self._dealer = self._init_dealer()

        if reanalyze and getattr(self, '_has_images', True) and 'od_raw' in vars(self):
            self.analyze_ods()
        elif not getattr(self, '_has_images', True):
            self._clear_image_analysis_attrs()
        if refresh_statistics:
            self._refresh_repeat_statistics()
        return self

    def _flatten_structured_ndarray(self, arr, old_dims):
        arr = np.asarray(arr)
        if arr.ndim < len(old_dims):
            return arr
        if tuple(arr.shape[:len(old_dims)]) != tuple(old_dims):
            return arr
        return arr.reshape(int(np.prod(old_dims)), *arr.shape[len(old_dims):])

    @staticmethod
    def _flatten_padded_ndarray(arr, old_dims, valid_mask):
        arr = np.asarray(arr)
        if arr.ndim < len(old_dims):
            return arr
        if tuple(arr.shape[:len(old_dims)]) != tuple(old_dims):
            return arr
        return arr[np.asarray(valid_mask, dtype=bool)]

    def flatten_xvar(self, xvar_param_key_or_idx, *, reanalyze=True):
        """Treat one structured xvar as repeated shots on the adjacent axis.

        Supports two-axis vaults. For the monitored Rabi use case,
        ``set_xvar('amp_imaging')`` produces ``['amp_imaging', 'dummy']``;
        ``flatten_xvar('dummy')`` then removes ``dummy`` and leaves a one-axis
        vault whose ``amp_imaging`` xvar has repeated values. If the structured
        grid was padded, missing cells are dropped during flattening.
        """
        if int(getattr(self, 'Nvars', 0)) != 2:
            raise NotImplementedError(
                'AtomdataVault.flatten_xvar currently supports two-axis '
                'structured vaults only.'
            )

        if isinstance(xvar_param_key_or_idx, (int, np.integer)):
            flatten_idx = int(xvar_param_key_or_idx)
        else:
            try:
                flatten_idx = list(self.xvarnames).index(str(xvar_param_key_or_idx))
            except ValueError as e:
                raise KeyError(
                    f"Unknown xvar {xvar_param_key_or_idx!r}; available "
                    f"xvars are {list(self.xvarnames)}."
                ) from e
        if flatten_idx not in (0, 1):
            raise IndexError('xvar index is out of range for this vault.')

        keep_idx = 1 - flatten_idx
        old_dims = tuple(np.asarray(self.xvardims, dtype=int))
        padded_mask = getattr(self, '_padded_xvar_mask', None)
        if padded_mask is not None:
            valid_mask = np.asarray(padded_mask, dtype=bool)
            if valid_mask.shape != old_dims:
                raise ValueError(
                    'Padded xvar mask shape does not match xvardims: '
                    f'{valid_mask.shape} vs {old_dims}.'
                )

            if keep_idx == 0:
                keep_values = np.asarray(self.xvars[0])
                keep_grid = np.broadcast_to(
                    keep_values.reshape((-1, 1)), old_dims
                )
                new_xvar = keep_grid[valid_mask]
            else:
                keep_grid = np.asarray(self.xvars[1])
                if keep_grid.shape != old_dims:
                    keep_grid = np.broadcast_to(keep_grid, old_dims)
                new_xvar = keep_grid[valid_mask]

            if hasattr(self, 'shot_run_id'):
                self.shot_run_id = self._flatten_padded_ndarray(
                    self.shot_run_id, old_dims, valid_mask
                )

            for key, values in list(getattr(self, '_shot_param_values', {}).items()):
                self._shot_param_values[key] = self._flatten_padded_ndarray(
                    values, old_dims, valid_mask
                )

            for key in self.data.keys:
                value = vars(self.data)[key]
                if isinstance(value, np.ndarray):
                    vars(self.data)[key] = self._flatten_padded_ndarray(
                        value, old_dims, valid_mask
                    )

            if hasattr(self, 'scope_data'):
                for scope_key, ch_dict in self.scope_data.items():
                    for ch, trace in ch_dict.items():
                        trace.t = self._flatten_padded_ndarray(
                            trace.t, old_dims, valid_mask
                        )
                        trace.v = self._flatten_padded_ndarray(
                            trace.v, old_dims, valid_mask
                        )

            _skip = {
                'images', 'image_timestamps', 'shot_run_id', 'xvars', 'xvarnames',
                'xvardims', 'data', 'scope_data', 'params', 'p', 'camera_params',
                'run_info', 'roi', 'avg', 'std', 'sem', 'sort_idx', 'sort_N',
            }
            for key, val in list(vars(self).items()):
                if key in _skip or key.startswith('_'):
                    continue
                if isinstance(val, np.ndarray):
                    vars(self)[key] = self._flatten_padded_ndarray(
                        val, old_dims, valid_mask
                    )

            keep_name = self.xvarnames[keep_idx]
            self.xvarnames = [keep_name]
            self.xvars = [new_xvar]
            self.xvardims = np.array([new_xvar.shape[0]], dtype=int)
            self.Nvars = 1
            setattr(self.params, keep_name, new_xvar)
            self.params.N_repeats = 1
            self.params.N_shots_with_repeats = int(new_xvar.shape[0])
            if hasattr(self.params, 'N_shots'):
                self.params.N_shots = int(new_xvar.shape[0])

            self.sort_idx = np.array([])
            self.sort_N = np.array([])
            self._structured_xvars = False
            self._padded_xvar_mask = None
            self._dealer = self._init_dealer()

            if reanalyze and getattr(self, '_has_images', True) and 'od_raw' in vars(self):
                self.analyze_ods()
            elif not getattr(self, '_has_images', True):
                self._clear_image_analysis_attrs()
            self._refresh_repeat_statistics()
            return self

        keep_values = np.asarray(self.xvars[keep_idx])
        flatten_len = int(old_dims[flatten_idx])
        keep_len = int(old_dims[keep_idx])

        if flatten_idx == 1:
            new_xvar = np.repeat(keep_values, flatten_len)
        else:
            new_xvar = np.tile(keep_values, flatten_len)

        if hasattr(self, 'shot_run_id'):
            self.shot_run_id = self._flatten_structured_ndarray(
                self.shot_run_id, old_dims
            )

        for key, values in list(getattr(self, '_shot_param_values', {}).items()):
            values = np.asarray(values)
            self._shot_param_values[key] = self._flatten_structured_ndarray(
                values, old_dims
            )

        if self._has_images and np.asarray(self.images).size:
            self.images = self._flatten_structured_ndarray(self.images, old_dims)
            self.image_timestamps = self._flatten_structured_ndarray(
                self.image_timestamps, old_dims
            )

        for key in self.data.keys:
            value = vars(self.data)[key]
            if isinstance(value, np.ndarray):
                vars(self.data)[key] = self._flatten_structured_ndarray(value, old_dims)

        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    trace.t = self._flatten_structured_ndarray(trace.t, old_dims)
                    trace.v = self._flatten_structured_ndarray(trace.v, old_dims)

        _skip = {
            'images', 'image_timestamps', 'shot_run_id', 'xvars', 'xvarnames',
            'xvardims', 'data', 'scope_data', 'params', 'p', 'camera_params',
            'run_info', 'roi', 'avg', 'std', 'sem', 'sort_idx', 'sort_N',
        }
        for key, val in list(vars(self).items()):
            if key in _skip or key.startswith('_'):
                continue
            if isinstance(val, np.ndarray):
                vars(self)[key] = self._flatten_structured_ndarray(val, old_dims)

        keep_name = self.xvarnames[keep_idx]
        self.xvarnames = [keep_name]
        self.xvars = [new_xvar]
        self.xvardims = np.array([new_xvar.shape[0]], dtype=int)
        self.Nvars = 1
        setattr(self.params, keep_name, new_xvar)
        self.params.N_repeats = flatten_len if keep_idx == 0 else keep_len
        self.params.N_shots_with_repeats = int(new_xvar.shape[0])
        if hasattr(self.params, 'N_shots'):
            self.params.N_shots = int(len(np.unique(keep_values)))

        self.sort_idx = np.array([])
        self.sort_N = np.array([])
        self._structured_xvars = False
        self._dealer = self._init_dealer()

        if reanalyze and getattr(self, '_has_images', True) and 'od_raw' in vars(self):
            self.analyze_ods()
        elif not getattr(self, '_has_images', True):
            self._clear_image_analysis_attrs()
        self._refresh_repeat_statistics()
        return self

    # ------------------------------------------------------------------
    # Ragged-repeat-aware statistics (avg / std / sem by unique xvar value)
    # ------------------------------------------------------------------
    @staticmethod
    def _stat_skip_keys():
        """Attributes that must never be treated as reducible scan-shaped
        data when building or collapsing statistics."""
        return {
            'avg', 'std', 'sem',
            '_repeat_sem_source', '_repeat_sem_divisor',
            'params', 'p', 'camera_params', 'run_info', 'roi',
            'data', 'scope_data', '_analysis_tags', '_dealer',
            '_ds', 'server_talk',
            'shot_run_id', 'source_run_ids', 'source_repeat_counts',
            'param_disagreements', 'source_param_values', 'sort_idx', 'sort_N',
            'xvars', 'xvarnames', 'xvardims',
        }

    @staticmethod
    def _grouped_mean_std(arr, inverse, n_groups, counts):
        """NaN-aware group mean/std along axis 0 by ``inverse`` group labels.

        Returns population std (ddof=0), matching the base atomdata repeat
        statistics. NaN entries (e.g. from NaN-padded DataVault keys that are
        absent in some runs) are ignored per element.
        """
        arr = np.asarray(arr, dtype=np.float64)
        trailing = arr.shape[1:]
        shp = (n_groups,) + trailing
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        csum = np.zeros(shp, dtype=np.float64)
        sqsum = np.zeros(shp, dtype=np.float64)
        ncnt = np.zeros(shp, dtype=np.float64)
        np.add.at(csum, inverse, vals)
        np.add.at(sqsum, inverse, vals * vals)
        np.add.at(ncnt, inverse, finite.astype(np.float64))
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = csum / ncnt
            var = sqsum / ncnt - mean * mean
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)
        return mean, std

    @staticmethod
    def _sem_from_std(std, counts):
        n = np.asarray(counts, dtype=np.float64)
        shp = (std.shape[0],) + (1,) * (std.ndim - 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return std / np.sqrt(n).reshape(shp)

    def _copy_metadata_to_ragged_sibling(self, ad_out, unique_xvar):
        """Populate a stat-sibling object (avg/std/sem) with metadata whose
        single scan axis is the *unique* xvar values."""
        ad_out._lite = self._lite
        ad_out.server_talk = self.server_talk
        ad_out._ds = getattr(self, '_ds', None)
        ad_out._dealer = None
        ad_out.images = self.images
        ad_out.image_timestamps = self.image_timestamps
        ad_out.experiment_code = getattr(self, 'experiment_code', None)

        ad_out.params = copy.deepcopy(self.params)
        ad_out.p = ad_out.params
        ad_out.camera_params = copy.deepcopy(self.camera_params)
        ad_out.run_info = copy.deepcopy(self.run_info)
        ad_out.roi = copy.deepcopy(self.roi)

        ad_out.xvarnames = list(self.xvarnames)
        ad_out.xvars = [np.array(unique_xvar, copy=True)]
        ad_out.Nvars = 1
        ad_out.xvardims = np.array([len(unique_xvar)], dtype=int)
        setattr(ad_out.params, ad_out.xvarnames[0], ad_out.xvars[0])
        ad_out.params.N_repeats = 1

        ad_out.sort_idx = np.array([])
        ad_out.sort_N = np.array([])

        ad_out.data = _VaultDataVault()
        ad_out.avg = None
        ad_out.std = None
        ad_out.sem = None
        ad_out._repeat_sem_source = None
        ad_out._repeat_sem_divisor = None
        ad_out._repeat_lazy_stat_context = None
        ad_out.source_run_ids = list(getattr(self, 'source_run_ids', []))

        ad_out._analysis_tags = analysis_tags(
            self._analysis_tags.roi_id, self._analysis_tags.imaging_type
        )
        ad_out._analysis_tags.xvars_shuffled = False

    def _build_grouped_statistics(self):
        """Build eager avg/std/sem siblings by grouping shots by unique xvar
        value. Handles ragged repeat counts (overlapping ranges) and uses each
        point's own count for SEM."""
        xvar = np.asarray(self.xvars[0])
        unique, inverse, counts = np.unique(
            xvar, return_inverse=True, return_counts=True
        )
        inverse = np.asarray(inverse).ravel()
        n_groups = unique.size

        ad_avg = object.__new__(self.__class__)
        ad_std = object.__new__(self.__class__)
        ad_sem = object.__new__(self.__class__)
        for sib in (ad_avg, ad_std, ad_sem):
            self._copy_metadata_to_ragged_sibling(sib, unique)

        skip = self._stat_skip_keys()

        def _reduce(value):
            mean, std = self._grouped_mean_std(value, inverse, n_groups, counts)
            sem = self._sem_from_std(std, counts)
            return mean, std, sem

        # Top-level scan-shaped arrays (od, od_raw, atom_number, fits, ...).
        for key, value in vars(self).items():
            if key in skip or key.startswith('_'):
                continue
            if self._is_scan_shaped_numeric_array(value):
                mean, std, sem = _reduce(value)
                vars(ad_avg)[key] = mean
                vars(ad_std)[key] = std
                vars(ad_sem)[key] = sem
            else:
                for sib in (ad_avg, ad_std, ad_sem):
                    if key not in vars(sib):
                        vars(sib)[key] = value

        # DataVault container.
        for key in self.data.keys:
            value = vars(self.data)[key]
            if self._is_scan_shaped_numeric_array(value):
                mean, std, sem = _reduce(value)
                vars(ad_avg.data)[key] = mean
                vars(ad_std.data)[key] = std
                vars(ad_sem.data)[key] = sem
            else:
                for sib in (ad_avg, ad_std, ad_sem):
                    vars(sib.data)[key] = value
            for sib in (ad_avg, ad_std, ad_sem):
                sib.data.keys.append(key)

        # Scope data (best effort; large arrays).
        if hasattr(self, 'scope_data'):
            from waxa.atomdata_base import ScopeTraceArray
            avg_scope, std_scope, sem_scope = {}, {}, {}
            try:
                for scope_key, ch_dict in self.scope_data.items():
                    avg_scope[scope_key] = {}
                    std_scope[scope_key] = {}
                    sem_scope[scope_key] = {}
                    for ch, trace in ch_dict.items():
                        out = {'t': {}, 'v': {}}
                        for ax in ('t', 'v'):
                            val = np.asarray(getattr(trace, ax))
                            if self._is_scan_shaped_numeric_array(val):
                                mean, std, sem = _reduce(val)
                            else:
                                mean = std = sem = val
                            out[ax] = (mean, std, sem)
                        avg_scope[scope_key][ch] = ScopeTraceArray(
                            scope_key, ch, out['t'][0], out['v'][0])
                        std_scope[scope_key][ch] = ScopeTraceArray(
                            scope_key, ch, out['t'][1], out['v'][1])
                        sem_scope[scope_key][ch] = ScopeTraceArray(
                            scope_key, ch, out['t'][2], out['v'][2])
                ad_avg.scope_data = avg_scope
                ad_std.scope_data = std_scope
                ad_sem.scope_data = sem_scope
            except Exception as e:
                warnings.warn(
                    f"AtomdataVault: failed to reduce scope_data statistics "
                    f"({e}); scope stats unavailable.",
                    stacklevel=2,
                )

        self.avg = ad_avg
        self.std = ad_std
        self.sem = ad_sem
        self._repeat_lazy_stat_context = None

    def _refresh_repeat_statistics(self):
        """Override: group by unique xvar value so overlapping ranges with
        ragged repeat counts average correctly."""
        if not self._merge_overlap or int(getattr(self, 'Nvars', 1)) != 1:
            return atomdata_base._refresh_repeat_statistics(self)
        self._build_grouped_statistics()

    # ------------------------------------------------------------------
    # Collapsing / provenance / auditing
    # ------------------------------------------------------------------
    def collapse_to_unique(self, reanalyze=True):
        """Permanently collapse the vault onto its unique xvar values, averaging
        all repeats/overlaps. Analogous to ``avg_repeats`` but ragged-safe.

        Parameters
        ----------
        reanalyze : bool
            If True (default) and the raw OD is available, re-run ``analyze_ods``
            so fits/atom-number are recomputed from the collapsed OD.
        """
        if getattr(self._analysis_tags, 'averaged', False):
            print('AtomdataVault is already collapsed to unique xvar values.')
            return self

        xvar = np.asarray(self.xvars[0])
        unique, inverse, counts = np.unique(
            xvar, return_inverse=True, return_counts=True
        )
        inverse = np.asarray(inverse).ravel()
        n_groups = unique.size

        skip = self._stat_skip_keys()
        for key, value in list(vars(self).items()):
            if key in skip or key.startswith('_'):
                continue
            if self._is_scan_shaped_numeric_array(value):
                mean, _ = self._grouped_mean_std(value, inverse, n_groups, counts)
                vars(self)[key] = mean

        for key in self.data.keys:
            value = vars(self.data)[key]
            if self._is_scan_shaped_numeric_array(value):
                mean, _ = self._grouped_mean_std(value, inverse, n_groups, counts)
                vars(self.data)[key] = mean

        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    for ax in ('t', 'v'):
                        val = np.asarray(getattr(trace, ax))
                        if self._is_scan_shaped_numeric_array(val):
                            mean, _ = self._grouped_mean_std(
                                val, inverse, n_groups, counts)
                            setattr(trace, ax, mean)

        # Collapse provenance to the set of runs contributing to each point.
        if hasattr(self, 'shot_run_id') and np.asarray(self.shot_run_id).dtype != object:
            rid = np.asarray(self.shot_run_id)
            self.shot_run_id = np.array(
                [np.unique(rid[inverse == g]) for g in range(n_groups)],
                dtype=object,
            )

        # Raw images no longer line up with the collapsed axis.
        if self._has_images:
            self.images = np.array([])
            self.image_timestamps = np.array([])

        self.xvars[0] = unique
        setattr(self.params, self.xvarnames[0], unique)
        self.xvardims = np.array([n_groups], dtype=int)
        self.params.N_repeats = 1
        self.params.N_shots_with_repeats = n_groups
        if hasattr(self.params, 'N_shots'):
            self.params.N_shots = n_groups
        self._analysis_tags.averaged = True

        if reanalyze and 'od_raw' in vars(self):
            self.analyze_ods()
        self._refresh_repeat_statistics()
        return self

    def shots_from_run(self, run_id):
        """Boolean mask selecting the shots that came from ``run_id``."""
        rid = np.asarray(self.shot_run_id)
        if rid.dtype == object and any(
                item is not None and np.asarray(item).ndim > 0
                for item in rid.ravel()):
            raise RuntimeError(
                'Per-shot provenance is unavailable after collapse_to_unique.'
            )
        return rid == int(run_id)

    def atomdata(self, run_id, *, copy_obj=False):
        """Return the already-loaded source ``atomdata`` for ``run_id``.

        The returned object is the materialized source chunk used to build this
        vault, so it reflects the vault's construction options (``lite``,
        ``ignore_images``, ROI reuse, and any construction-time unshuffle copy)
        without reloading from disk. Pass ``copy_obj=True`` to get a deep copy
        that can be mutated independently.
        """
        rid = int(run_id)
        try:
            ad = self._source_atomdata_by_run_id[rid]
        except KeyError as e:
            raise KeyError(
                f"Run ID {rid} is not part of this AtomdataVault. Available "
                f"run IDs: {list(self.source_run_ids)}."
            ) from e
        if copy_obj:
            return copy.deepcopy(ad)
        return ad

    def drop_runs(self, run_ids, reanalyze=True):
        """Remove every shot belonging to ``run_ids`` and refresh statistics.

        Useful for excluding an outlier/aborted run discovered after loading.
        """
        if np.isscalar(run_ids):
            run_ids = [run_ids]
        drop = {int(r) for r in run_ids}

        if not hasattr(self, 'shot_run_id'):
            raise RuntimeError('shot_run_id provenance is unavailable.')
        rid = np.asarray(self.shot_run_id)
        if rid.ndim != 1:
            raise RuntimeError(
                'drop_runs is only available on flat, unstructured vaults.'
            )
        if rid.dtype == object and any(
                item is not None and np.asarray(item).ndim > 0
                for item in rid.ravel()):
            raise RuntimeError(
                'drop_runs is unavailable after collapse_to_unique.'
            )

        keep = np.where(~np.isin(rid, list(drop)))[0]
        if keep.size == rid.size:
            warnings.warn(
                'AtomdataVault.drop_runs: no shots matched the requested run '
                f'ids {sorted(drop)}.',
                stacklevel=2,
            )
            return self
        if keep.size == 0:
            raise ValueError('drop_runs would remove every shot in the vault.')

        self._reorder_shots(keep)
        self.source_run_ids = [r for r in self.source_run_ids if r not in drop]
        for rid_to_drop in drop:
            self._source_atomdata_by_run_id.pop(rid_to_drop, None)
        self.run_info.run_id = list(self.source_run_ids)

        if reanalyze and 'od_raw' in vars(self):
            self.analyze_ods()
        self._refresh_repeat_statistics()
        return self

    @staticmethod
    def _fmt_param_value(value):
        if value is None:
            return 'NA'
        arr = np.asarray(value)
        if arr.ndim == 0:
            try:
                return f'{float(arr):.6g}'
            except (TypeError, ValueError):
                return str(value)
        if arr.size <= 4 and np.issubdtype(arr.dtype, np.number):
            return '[' + ', '.join(f'{v:.4g}' for v in arr.ravel()) + ']'
        return f'<{arr.dtype} array shape {arr.shape}>'

    def param_report(self):
        """Print a per-run summary of the source runs and any fixed-parameter
        disagreements. Returns the disagreements dict."""
        rids = list(self.source_run_ids)
        lines = [f'AtomdataVault: {len(rids)} source run(s)']
        lines.append('  run_ids: ' + ', '.join(str(r) for r in rids))

        rc = getattr(self, 'source_repeat_counts', None)
        if rc:
            lines.append('  N_repeats: '
                         + ', '.join(f'{r}:{rc.get(r, "?")}' for r in rids))

        if not self.param_disagreements:
            lines.append('  All fixed parameters agree across runs.')
        else:
            lines.append(f'  {len(self.param_disagreements)} fixed parameter(s) '
                         f'disagree across runs:')
            for key in sorted(self.param_disagreements):
                per_run = self.param_disagreements[key]
                vals = ', '.join(
                    f'{r}={self._fmt_param_value(per_run.get(r))}' for r in rids
                )
                lines.append(f'    {key}: {vals}')

        print('\n'.join(lines))
        return self.param_disagreements

    def add_runs(self, inputs, **overrides):
        """Return a NEW vault built from this vault's source runs plus
        ``inputs`` (run-ids or atomdata objects). Construction options are
        inherited from this vault unless overridden via keyword arguments."""
        kwargs = dict(self._build_kwargs)
        kwargs.update(overrides)
        combined = list(self.source_run_ids) + _flatten_inputs(inputs)
        return AtomdataVault(combined, **kwargs)

    # ------------------------------------------------------------------
    # Builder-aware discovery constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_run_range(cls, start_id, stop_id, experiment_name=None,
                       skip_missing=True, roi_id=None, lite=True, **kwargs):
        """Build a vault from a contiguous run-id range ``[start_id, stop_id]``.

        Missing/aborted run-ids are skipped (``skip_missing``). If
        ``experiment_name`` is given, only runs whose experiment class or
        filepath contains that substring are kept -- handy for an experiment
        builder that interleaves several experiment types. Subsequent runs
        reuse the first loaded run's ROI so the selector opens at most once.
        """
        start_id, stop_id = int(start_id), int(stop_id)
        if stop_id < start_id:
            start_id, stop_id = stop_id, start_id

        ads, skipped = [], []
        anchor_roi = roi_id
        for rid in range(start_id, stop_id + 1):
            try:
                ad = atomdata(rid, roi_id=anchor_roi, lite=lite)
            except Exception:
                if skip_missing:
                    skipped.append(rid)
                    continue
                raise
            if anchor_roi is None:
                anchor_roi = int(ad.run_info.run_id)
                try:
                    ad.save_roi_h5()
                except Exception:
                    pass
            if experiment_name is not None:
                name = str(getattr(ad.run_info, 'expt_class', '') or '')
                fpath = str(getattr(ad.run_info, 'experiment_filepath', '') or '')
                target = experiment_name.lower()
                if target not in name.lower() and target not in fpath.lower():
                    skipped.append(rid)
                    continue
            ads.append(ad)

        if not ads:
            raise ValueError(
                f'No loadable runs found in range [{start_id}, {stop_id}]'
                + (f' matching experiment_name={experiment_name!r}'
                   if experiment_name else '') + '.'
            )
        if skipped:
            preview = ', '.join(str(r) for r in skipped[:20])
            more = '...' if len(skipped) > 20 else ''
            warnings.warn(
                f'AtomdataVault.from_run_range: skipped {len(skipped)} run(s) '
                f'({preview}{more}).',
                stacklevel=2,
            )
        return cls(ads, roi_id=roi_id, lite=lite, **kwargs)

    @classmethod
    def from_builder(cls, start_id, stop_id, experiment_name, **kwargs):
        """Convenience wrapper around :meth:`from_run_range` that requires an
        ``experiment_name`` filter -- the typical experiment-builder case where
        a run-id range contains the builder's runs (possibly interleaved with
        others)."""
        return cls.from_run_range(
            start_id, stop_id, experiment_name=experiment_name, **kwargs
        )

    # ------------------------------------------------------------------
    # ROI / recrop
    # ------------------------------------------------------------------
    def recrop(self, roi_id=None, use_saved=False):
        """Select a new ROI and re-crop every concatenated run at once.

        Because the vault stores the concatenated ``od_raw`` for all runs on
        a single leading axis, cropping with one ROI naturally re-crops every
        run identically.

        Args:
            roi_id (None, int, or str): See ``atomdata.recrop``. If None,
                prompts the ROI selection GUI (unless a saved ROI is used).
            use_saved (bool): If False (default), ignores any saved ROI and
                forces selection of a new one.
        """
        if not getattr(self, '_has_images', True):
            print("no images in dataset (ignore_images), no roi to crop")
            return
        if self._lite:
            raise NotImplementedError(
                "recrop is not supported on a lite AtomdataVault; the lite "
                "runs are already cropped. Build the vault from full "
                "(non-lite) runs to recrop all runs."
            )
        # od_raw is the concatenation of every run's raw ODs; selecting one
        # ROI here and re-running analyze_ods re-crops all runs together.
        od_flat = self.od_raw.reshape(-1, *self.od_raw.shape[-2:])
        self.roi.load_roi(roi_id, use_saved, display_ods=od_flat)
        self.analyze_ods()
        self._refresh_repeat_statistics()

    # ------------------------------------------------------------------
    # Unsupported operations
    # ------------------------------------------------------------------
    def reshuffle(self):
        raise NotImplementedError(
            "reshuffle is not supported on AtomdataVault."
        )

    def unshuffle(self, reanalyze=True):
        raise NotImplementedError(
            "unshuffle is not supported on AtomdataVault "
            "(vault is constructed unshuffled)."
        )

    def reassign_repeats(self, xvar_idx):
        raise NotImplementedError(
            "reassign_repeats is not supported on AtomdataVault."
        )

    def transpose_data(self, new_xvar_idx=[], reanalyze=True):
        raise NotImplementedError(
            "transpose_data is not supported on AtomdataVault (single axis)."
        )

    def _unshuffle_old_data(self):
        # Vault construction already places everything in unshuffled order.
        return
