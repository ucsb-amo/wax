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

    * Multi-axis runs. Runs with ``Nvars > 1`` are *stacked* along a per-run
      parameter (``promote_xvar``), giving an ``Nvars + 1`` axis vault, e.g.
      two (phase x detuning) runs at different trap compressions become a
      (compression, phase, detuning) dataset. An inner axis the runs sample
      differently is merged onto the sorted union of its values; cells a run
      did not take are NaN and ``vault.stack_mask`` marks the real ones. The
      runs' own analysis is reused, so nothing is re-cropped or re-fit.
    * Axis relabelling. ``remap_xvar`` replaces an xvar by a function of it
      (photodiode volts -> optical power) or by an existing param of the same
      shape; ``data_container_to_xvar`` does the same from a recorded
      ``vault.data`` key. Both add the new values to ``params``, keep the old
      param, leave the data layout alone, and work on flat and stacked vaults.

Limitations:
    * 1-D inputs concatenate; multi-axis inputs stack (one run per promoted
      value, identical xvarnames). The two cannot be mixed, and
      ``set_xvar`` / ``flatten_xvar`` / ``drop_runs`` / ``collapse_to_unique``
      / ``recrop`` are unavailable on a stacked vault.
    * 1-D inputs must share the same ``xvarnames[0]`` unless
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
    _collapse_shared_time_axes,
    REPEAT_STAT_LAZY_BYTES,
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
    plus arbitrary array attributes).

    When constructed with a source vault and a stat kind it also backs the
    avg/std/sem siblings: keys added to the parent's DataVault after the
    siblings were built are grouped on demand by ``__getattr__``. See
    ``waxa.atomdata_base._RepeatDataVault``.
    """

    def __init__(self, source=None, kind=None):
        self._keys = []
        self._source = source
        self._kind = kind

    @property
    def keys(self):
        if self._source is None:
            return self._keys
        return list(dict.fromkeys(self._keys + list(self._source.data.keys)))

    @keys.setter
    def keys(self, value):
        self._keys = list(value)

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        source = self.__dict__.get('_source')
        kind = self.__dict__.get('_kind')
        if source is None or kind is None:
            raise AttributeError(key)
        value = getattr(source.data, key)
        if not source._is_scan_shaped_numeric_array(value):
            return value
        return source._grouped_array_stats(value)[
            {'mean': 0, 'std': 1, 'sem': 2}[kind]
        ]


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
    regenerate_lite : bool
        If True, force lite loading (implies ``lite=True``) and regenerate every
        pre-cropped lite file from the full run cropped to the anchor ROI,
        overwriting any existing lite copy instead of reusing it. Use this when
        stale lite files were baked with a different ROI (e.g. a full frame),
        which otherwise raises an image-shape mismatch at concatenation. Defaults
        to False.
    uniform_roi : bool
        If True (default), the ROI is resolved once from the first input and
        reused for every subsequent run so a single, consistent crop is applied
        across the whole vault. Subsequent non-lite runs load with the anchor
        ``roi_id``; subsequent lite runs reuse an existing pre-cropped lite file
        when one is present (a warning is emitted, since its baked ROI is
        trusted rather than re-applied), otherwise the lite file is generated
        from the full run cropped to the anchor ROI. If False, each run is
        loaded independently with the caller-supplied ``roi_id`` (the legacy
        per-run behavior, where a ``roi_id=None`` GUI can open once per run).
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
    decimate_scope_data : int or None
        Forwarded to every ``atomdata(...)`` load: keep only every
        ``decimate_scope_data``-th sample of each scope trace (t and v) to
        save memory. A value of 0 (or None) loads all samples with no
        decimation or averaging.
    smooth_decimate : bool
        Only used when ``decimate_scope_data`` is set. If True (default),
        each group of ``decimate_scope_data`` samples is block-averaged;
        if False, plain stride sampling keeps every
        ``decimate_scope_data``-th sample.
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
        For multi-axis input runs this is the axis the runs are stacked
        along (required unless exactly one scalar parameter differs).
    flatten_xvar : str, int, or None
        Structured xvar key/index to flatten automatically after any promotion.
    skip_missing : bool
        If True (default), run-id inputs that fail to load because the run is
        missing/aborted are skipped with a warning and loading continues.
        Non-missing failures are still raised.
    """

    def __init__(self,
                 inputs,
                 roi_id=None,
                 lite=False,
                 regenerate_lite=False,
                 uniform_roi=True,
                 xvarname_override=False,
                 sort=True,
                 merge_overlap=True,
                 drop_raw_images=False,
                 auto_lite_threshold=8,
                 ignore_images=False,
                 decimate_scope_data=None,
                 smooth_decimate=True,
                 scope_merge='pad_nan',
                 structure='auto',
                 xvar_mode='pad',
                 promote_xvar=None,
                 flatten_xvar=None,
                 skip_missing=True):

        # regenerate_lite forces lite loading and re-crops every lite file to
        # the anchor ROI (see _load_lite_with_anchor), overriding the default
        # reuse of existing pre-cropped lite copies.
        self._regenerate_lite = bool(regenerate_lite)
        if self._regenerate_lite:
            lite = True

        # Lightweight book-keeping expected by inherited helpers.
        self._lite = lite
        self._ignore_images = bool(ignore_images)
        self._timing_enabled = False
        self._timing = {}
        self.server_talk = None

        # Vault-specific configuration.
        self._merge_overlap = bool(merge_overlap)
        self._uniform_roi = bool(uniform_roi)
        self._drop_raw_images = bool(drop_raw_images)
        self._skip_missing = bool(skip_missing)
        if scope_merge not in ('strict', 'pad_nan', 'skip'):
            raise ValueError(
                "scope_merge must be one of 'strict', 'pad_nan', or 'skip'."
            )
        self._scope_merge = scope_merge
        if decimate_scope_data is not None:
            decimate_scope_data = int(decimate_scope_data)
            if decimate_scope_data < 0:
                raise ValueError("decimate_scope_data must be a non-negative integer.")
            if decimate_scope_data == 0:
                # 0 means "load everything" — same as None.
                decimate_scope_data = None
        self._decimate_scope_data = decimate_scope_data
        self._smooth_decimate = bool(smooth_decimate)
        # Forwarded to every atomdata(...) load below.
        self._scope_load_kwargs = dict(
            decimate_scope_data=decimate_scope_data,
            smooth_decimate=bool(smooth_decimate),
        )
        self._structure = structure
        self._xvar_mode = xvar_mode
        # Kwargs needed to rebuild an equivalent vault (used by add_runs).
        self._build_kwargs = dict(
            roi_id=roi_id,
            lite=lite,
            regenerate_lite=regenerate_lite,
            uniform_roi=uniform_roi,
            xvarname_override=xvarname_override,
            sort=sort,
            merge_overlap=merge_overlap,
            drop_raw_images=drop_raw_images,
            auto_lite_threshold=auto_lite_threshold,
            ignore_images=ignore_images,
            decimate_scope_data=decimate_scope_data,
            smooth_decimate=smooth_decimate,
            scope_merge=scope_merge,
            structure=structure,
            xvar_mode=xvar_mode,
            promote_xvar=promote_xvar,
            flatten_xvar=flatten_xvar,
            skip_missing=skip_missing,
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

        # Load run-ids sequentially. With uniform_roi (default), the ROI is
        # resolved once from the first input and reused for every subsequent
        # run so a single, consistent crop is applied across the whole vault.
        ads = self._materialize_inputs(
            raw_inputs, roi_id, lite, self._ignore_images, self._uniform_roi,
        )
        if len(ads) == 0:
            raise ValueError(
                "AtomdataVault: no loadable inputs remain after skipping "
                "missing run-ids."
            )

        # 2. Validate compatibility. N_repeats may differ across inputs when
        #    merge_overlap is on (grouped statistics handle ragged counts).
        #    Multi-axis runs take the stacked path (see _assemble_stacked).
        self._stacked = int(getattr(ads[0], 'Nvars', 0)) > 1
        if self._stacked:
            self._validate_inputs_nd(ads, ignore_images=self._ignore_images)
        else:
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
                # The stacked path reuses each chunk's own analysis, so it has
                # to be redone in the unshuffled order.
                ad_copy.unshuffle(reanalyze=self._stacked)
                if not self._stacked and getattr(ad_copy, '_has_images', True):
                    ad_copy._sort_images()
                chunks.append(ad_copy)
            else:
                chunks.append(ad)

        if self._stacked:
            if flatten_xvar is not None:
                raise NotImplementedError(
                    'flatten_xvar is not supported for multi-axis input runs.'
                )
            self._assemble_stacked(chunks, promote_xvar, structure, sort, roi_id)
            return

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
    # Input materialization / ROI anchoring
    # ------------------------------------------------------------------
    def _materialize_inputs(self, raw_inputs, roi_id, lite, ignore_images,
                            uniform_roi):
        """Load every input into an ``atomdata`` object.

        When ``uniform_roi`` is True the ROI is resolved once (from the first
        input) and reused for every subsequent run: subsequent non-lite runs
        load with ``roi_id=anchor``; subsequent lite runs reuse an existing
        pre-cropped lite file when present (with a warning), otherwise the lite
        file is generated from the full run cropped to the anchor ROI. When
        False, each run is loaded independently with the caller-supplied
        ``roi_id`` (the legacy per-run behavior).
        """
        # A server_talk instance is only needed to probe for / generate lite
        # copies when anchoring the ROI on lite runs.
        server_talk = None
        if uniform_roi and lite and not ignore_images:
            from waxa.data.server_talk import server_talk as _st
            server_talk = _st()

        # The anchor ROI only needs persisting to the run's h5 (so subsequent
        # runs can look it up, and so lite copies can be generated from it) when
        # there is more than one run or lite data is involved.
        has_subsequent_int_loads = any(
            isinstance(it, (int, np.integer)) for it in raw_inputs[1:]
        )
        persist_anchor = has_subsequent_int_loads or lite

        ads = []
        skipped_missing = []
        anchor_roi_id = None      # int run-id, str key, or None
        anchor_established = False

        for item in raw_inputs:
            if isinstance(item, atomdata_base):
                if uniform_roi and not anchor_established and not ignore_images:
                    if getattr(item, 'roi', None) is not None:
                        if persist_anchor:
                            item.save_roi_h5()
                        anchor_roi_id = int(item.run_info.run_id)
                    anchor_established = True
                ads.append(item)
                continue

            if not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"AtomdataVault inputs must be atomdata objects or "
                    f"run_id ints, got {type(item).__name__}."
                )

            rid = int(item)

            try:
                if not uniform_roi:
                    # Legacy per-run behavior: independent ROI per run.
                    ads.append(atomdata(rid, roi_id=roi_id, lite=lite,
                                        ignore_images=ignore_images,
                                        **self._scope_load_kwargs))
                    continue

                if not anchor_established:
                    ad, anchor_roi_id = self._load_anchor_run(
                        rid, roi_id, lite, ignore_images, server_talk,
                        persist_anchor,
                    )
                    anchor_established = True
                    ads.append(ad)
                    continue

                ads.append(self._load_with_anchor(
                    rid, anchor_roi_id, lite, ignore_images, server_talk,
                ))
            except Exception as e:
                if self._skip_missing and self._is_missing_run_error(e):
                    skipped_missing.append(rid)
                    continue
                raise

        if skipped_missing:
            preview = ', '.join(str(r) for r in skipped_missing[:20])
            more = '...' if len(skipped_missing) > 20 else ''
            warnings.warn(
                f'AtomdataVault: skipped {len(skipped_missing)} missing '
                f'run-id(s) while loading inputs ({preview}{more}).',
                stacklevel=2,
            )

        return ads

    @staticmethod
    def _is_missing_run_error(exc):
        """Best-effort check for run-id missing/aborted load failures."""
        if isinstance(exc, FileNotFoundError):
            return True
        msg = str(exc).lower()
        return any(token in msg for token in (
            'missing',
            'not found',
            'no such file',
            'could not find',
            'run id',
            'run_id',
            'data file',
        ))

    def _load_anchor_run(self, rid, roi_id, lite, ignore_images, server_talk,
                         persist_anchor=True):
        """Load the first run and establish the anchor ROI.

        Returns ``(atomdata, anchor_roi_id)`` where ``anchor_roi_id`` is an int
        run-id, a str ROI key, or ``None`` (images ignored).
        """
        if ignore_images:
            return atomdata(rid, roi_id=roi_id, lite=lite,
                            ignore_images=True,
                            **self._scope_load_kwargs), None

        if not lite:
            # Full load: honor the caller's roi_id (may open the GUI once when
            # None), then persist the resolved ROI so subsequent runs can look
            # it up by run-id.
            ad = atomdata(rid, roi_id=roi_id, lite=False,
                          **self._scope_load_kwargs)
            if persist_anchor:
                ad.save_roi_h5()
            return ad, (roi_id if roi_id is not None else rid)

        # lite=True: the anchor ROI must be known in full-frame coordinates so
        # it can crop the other runs. Resolve it cheaply when possible, falling
        # back to a full load + GUI selection on the first run only.
        full_ad = None
        if roi_id is not None:
            anchor_roi_id = roi_id
        elif self._saved_roi_in_regular_h5(rid, server_talk):
            anchor_roi_id = rid
        elif not self._regenerate_lite and self._lite_copy_exists(rid, server_talk):
            # An existing lite copy carries its own baked ROI, so there is
            # nothing to select: do not full-load the run just to open the ROI
            # GUI. The full-frame anchor is then only resolved if a later run
            # has no lite copy and must be cropped to it.
            anchor_roi_id = rid
        else:
            full_ad = atomdata(rid, roi_id=None, lite=False,
                               **self._scope_load_kwargs)
            full_ad.save_roi_h5()
            anchor_roi_id = rid

        ad = self._load_lite_with_anchor(
            rid, anchor_roi_id, server_talk, full_ad=full_ad,
        )
        return ad, anchor_roi_id

    def _load_with_anchor(self, rid, anchor_roi_id, lite, ignore_images,
                          server_talk):
        """Load a subsequent run reusing the anchor ROI."""
        if ignore_images:
            return atomdata(rid, lite=lite, ignore_images=True,
                            **self._scope_load_kwargs)
        if not lite:
            return atomdata(rid, roi_id=anchor_roi_id, lite=False,
                            **self._scope_load_kwargs)
        return self._load_lite_with_anchor(rid, anchor_roi_id, server_talk)

    def _load_lite_with_anchor(self, rid, anchor_roi_id, server_talk,
                               full_ad=None):
        """Return a lite ``atomdata`` for ``rid`` cropped to the anchor ROI.

        Reuses an existing pre-cropped lite file when present (warning that its
        baked ROI is trusted, not re-applied); otherwise generates the lite
        file from the full run cropped to the anchor ROI, reusing ``full_ad``
        when it has already been loaded. When ``self._regenerate_lite`` is set,
        the reuse path is skipped and the lite file is always regenerated
        (overwriting any existing copy) so it is cropped to the anchor ROI.
        """
        if not self._regenerate_lite and self._lite_copy_exists(rid, server_talk):
            warnings.warn(
                f"AtomdataVault: reusing the ROI baked into the existing lite "
                f"data for run {rid}; it is trusted as-is and may differ from "
                f"the anchor ROI (run {anchor_roi_id}). Delete the lite file to "
                f"regenerate it cropped to the anchor ROI.",
                stacklevel=2,
            )
            return atomdata(rid, lite=True, **self._scope_load_kwargs)

        gen = full_ad if full_ad is not None else atomdata(
            rid, roi_id=anchor_roi_id, lite=False,
            **self._scope_load_kwargs,
        )
        # save_lite_copy copies scope data from the source h5 file, so the
        # lite file always holds the full (undecimated) traces.
        gen.save_lite_copy(roi_id=anchor_roi_id, use_saved_roi=True)
        return atomdata(rid, lite=True, **self._scope_load_kwargs)

    @staticmethod
    def _lite_copy_exists(run_id, server_talk):
        """True if a pre-cropped lite HDF5 already exists for ``run_id``."""
        if server_talk is None:
            return False
        try:
            path = server_talk.find_data_file_by_run_id(
                int(run_id), lite=True, raise_on_missing=False, skip_check=True,
            )
        except Exception:
            return False
        return path is not None

    @staticmethod
    def _saved_roi_in_regular_h5(run_id, server_talk):
        """True if the regular (non-lite) h5 for ``run_id`` has a saved ROI.

        Reads only the file-level ROI attributes, so it never triggers the ROI
        selection GUI.
        """
        if server_talk is None:
            return False
        import h5py
        try:
            fpath, _ = server_talk.get_data_file(int(run_id), lite=False)
            with h5py.File(fpath, 'r') as f:
                return ('roix' in f.attrs) and ('roiy' in f.attrs)
        except Exception:
            return False

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

    # ------------------------------------------------------------------
    # Multi-axis inputs: stack along a promoted parameter
    # ------------------------------------------------------------------
    def _validate_inputs_nd(self, ads, ignore_images=False):
        first = ads[0]
        nvars = int(first.Nvars)
        names = [_decode_xvarname(n) for n in first.xvarnames]
        for ad in ads[1:]:
            rid = ad.run_info.run_id
            if int(getattr(ad, 'Nvars', 0)) != nvars:
                raise ValueError(
                    f"Nvars mismatch: run {rid} has Nvars={ad.Nvars} but the "
                    f"first input has Nvars={nvars}."
                )
            these = [_decode_xvarname(n) for n in ad.xvarnames]
            if these != names:
                raise ValueError(
                    f"xvarname mismatch: run {rid} scans {these} but the first "
                    f"input scans {names}. Multi-axis inputs must share their "
                    "xvars, in the same order."
                )
            if ad.run_info.imaging_type != first.run_info.imaging_type:
                raise ValueError(f"imaging_type mismatch on run {rid}.")
        self.source_repeat_counts = {
            int(ad.run_info.run_id): ad.params.N_repeats for ad in ads
        }

    def _resolve_stack_key(self, chunks, promote_xvar, structure):
        """The scalar per-run parameter that becomes xvar 0 of a stacked vault."""
        scalar_keys = sorted(
            key for key, per_run in self.param_disagreements.items()
            if all(np.ndim(v) == 0 and v is not None for v in per_run.values())
        )
        if promote_xvar is None:
            if structure == 'auto' and len(scalar_keys) == 1:
                promote_xvar = scalar_keys[0]
            else:
                raise ValueError(
                    "Multi-axis runs are combined by stacking them along a "
                    "per-run parameter: pass promote_xvar=<param key>. Scalar "
                    "parameters that differ across these runs: "
                    + (", ".join(scalar_keys) or "none") + "."
                )
        values = []
        for c in chunks:
            value = vars(c.params).get(promote_xvar, None)
            if value is None or np.ndim(value) != 0:
                raise KeyError(
                    f"promote_xvar={promote_xvar!r} is not a scalar parameter "
                    f"of run {c.run_info.run_id}."
                )
            values.append(np.asarray(value).item())
        if len(set(values)) != len(values):
            raise NotImplementedError(
                f"Several multi-axis runs share {promote_xvar!r} = "
                f"{sorted(values)}; merging runs within one promoted value is "
                "not implemented. Pass one run per value."
            )
        return str(promote_xvar), values

    @staticmethod
    def _union_axis(chunk_values):
        """One 1-D axis holding every chunk's values, and each chunk's index
        into it.

        Identical axes pass through untouched, repeats included. Axes that
        differ are merged onto their sorted union, which requires each chunk
        to sample a value once: a ragged axis cannot also carry the repeats.
        """
        arrays = [np.asarray(v) for v in chunk_values]
        if all(a.shape == arrays[0].shape and np.array_equal(a, arrays[0])
               for a in arrays[1:]):
            idx = np.arange(arrays[0].shape[0])
            return np.array(arrays[0], copy=True), [idx for _ in arrays]
        for a in arrays:
            if np.unique(a).size != a.size:
                raise NotImplementedError(
                    "An xvar that differs between runs cannot also carry the "
                    "repeats; put the repeats on an axis the runs share."
                )
        union = np.unique(np.concatenate(arrays))
        return union, [np.searchsorted(union, a) for a in arrays]

    @staticmethod
    def _stack_scan_arrays(arrays, inner_dims, index_maps):
        """Stack per-chunk scan-shaped arrays along a new axis 0, scattering
        each onto the (possibly larger) union grid. Cells a chunk did not
        sample are NaN (numeric, promoted to float64) or None (object)."""
        arrays = [np.asarray(a) for a in arrays]
        nv = len(inner_dims)
        trailing = arrays[0].shape[nv:]
        padded = any(tuple(a.shape[:nv]) != tuple(inner_dims) for a in arrays)
        if not padded:
            return np.stack(arrays, axis=0)
        if arrays[0].dtype == object:
            out = np.full((len(arrays), *inner_dims, *trailing), None, dtype=object)
        elif np.issubdtype(arrays[0].dtype, np.number):
            out = np.full((len(arrays), *inner_dims, *trailing), np.nan, dtype=np.float64)
        else:
            raise TypeError(f'cannot pad dtype {arrays[0].dtype}')
        for i, (a, maps) in enumerate(zip(arrays, index_maps)):
            out[(i, *np.ix_(*maps))] = a
        return out

    def _assemble_stacked(self, chunks, promote_xvar, structure, sort, roi_id):
        """Build an (Nvars + 1)-axis vault from multi-axis runs.

        Each run becomes one slab along a new leading axis, the promoted
        per-run parameter. Unlike the 1-D path nothing is re-analyzed: the
        scan-shaped results of every chunk (``od``, ``atom_number``, fit
        arrays, DataVault fields, ...) are stacked as they are, so the runs
        keep the ROI they were loaded with. ``vault.stack_mask`` marks the
        cells that hold data; the rest are NaN.
        """
        first = chunks[0]
        nv = int(first.Nvars)

        self.params = copy.deepcopy(first.params)
        self.p = self.params
        self.camera_params = copy.deepcopy(first.camera_params)
        self.run_info = copy.deepcopy(first.run_info)
        self.experiment_code = getattr(first, 'experiment_code', None)
        self._has_images = (
            False if self._ignore_images
            else bool(getattr(first, '_has_images', True))
        )

        self._warn_param_mismatches(chunks)
        key, values = self._resolve_stack_key(chunks, promote_xvar, structure)
        if sort:
            order = np.argsort(values, kind='stable')
            chunks = [chunks[i] for i in order]
            values = [values[i] for i in order]

        self.source_run_ids = [int(c.run_info.run_id) for c in chunks]
        self._source_atomdata_by_run_id = {
            int(c.run_info.run_id): c for c in chunks
        }
        self.run_info.run_id = list(self.source_run_ids)

        inner_names = [_decode_xvarname(n) for n in first.xvarnames]
        inner_xvars, per_axis_maps = [], []
        for j in range(nv):
            axis, maps = self._union_axis([c.xvars[j] for c in chunks])
            inner_xvars.append(axis)
            per_axis_maps.append(maps)
        inner_dims = tuple(len(x) for x in inner_xvars)
        # index_maps[i][j]: where chunk i's axis-j samples land on the union axis
        index_maps = [[per_axis_maps[j][i] for j in range(nv)]
                      for i in range(len(chunks))]

        def _scan_shaped(c, arr):
            return (isinstance(arr, np.ndarray) and arr.ndim >= nv
                    and tuple(arr.shape[:nv]) == tuple(c.xvardims))

        def _stack(getter, label, skipped):
            arrays = [getter(c) for c in chunks]
            if not all(_scan_shaped(c, a) for c, a in zip(chunks, arrays)):
                return None
            if len({a.shape[nv:] for a in arrays}) != 1:
                skipped.append(label)
                return None
            try:
                return self._stack_scan_arrays(arrays, inner_dims, index_maps)
            except TypeError:
                skipped.append(label)
                return None

        skip = {'images', 'image_timestamps', 'xvars', 'xvardims', 'sort_idx',
                'sort_N', 'avg', 'std', 'sem', 'od_raw'}
        if self._drop_raw_images or not self._has_images:
            skip |= {'img_atoms', 'img_light', 'img_dark'}
        skipped = []
        for attr, val in list(vars(first).items()):
            if attr.startswith('_') or attr in skip or not _scan_shaped(first, val):
                continue
            stacked = _stack(lambda c, a=attr: vars(c).get(a, None), attr, skipped)
            if stacked is not None:
                vars(self)[attr] = stacked

        self.data = _VaultDataVault()
        for k in first.data.keys:
            stacked = _stack(lambda c, k=k: vars(c.data).get(k, None), f'data.{k}', skipped)
            if stacked is not None:
                vars(self.data)[k] = stacked
                self.data.keys.append(k)
        if skipped:
            warnings.warn(
                "AtomdataVault: not stacked because their per-shot shapes "
                "differ between runs (different ROIs?): " + ", ".join(skipped),
                stacklevel=3,
            )
        if any(getattr(c, 'scope_data', None) for c in chunks):
            warnings.warn(
                "AtomdataVault: scope_data is not carried into a stacked "
                "(multi-axis) vault; read it from vault.atomdata(run_id).",
                stacklevel=3,
            )

        self.stack_mask = self._stack_scan_arrays(
            [np.ones(tuple(c.xvardims), dtype=float) for c in chunks],
            inner_dims, index_maps) == 1.0
        self._padded_xvar_mask = self.stack_mask
        self.shot_run_id = np.where(
            self.stack_mask,
            np.asarray(self.source_run_ids, dtype=np.int64).reshape(-1, *([1] * nv)),
            -1,
        )
        self._shot_param_values = {}

        self.images = np.array([])
        self.image_timestamps = np.array([])

        self.xvarnames = [key, *inner_names]
        self.xvars = [np.asarray(values), *inner_xvars]
        self.xvardims = np.array([len(x) for x in self.xvars], dtype=int)
        self.Nvars = nv + 1
        for name, axis in zip(self.xvarnames, self.xvars):
            setattr(self.params, name, axis)

        n_repeats = []
        for axis in self.xvars:
            _, counts = np.unique(axis, return_counts=True)
            n_repeats.append(int(counts.max()))
        self.params.N_repeats = np.array(n_repeats, dtype=int)
        self.params.N_shots_with_repeats = int(np.count_nonzero(self.stack_mask))
        if hasattr(self.params, 'N_shots'):
            self.params.N_shots = int(
                self.params.N_shots_with_repeats // max(int(np.prod(n_repeats)), 1)
            )

        self.sort_idx = np.array([])
        self.sort_N = np.array([])
        self._structured_xvars = True

        from waxa.data.data_saver import DataSaver
        self._ds = DataSaver()
        self._dealer = None
        self._analysis_tags = analysis_tags(
            roi_id=roi_id, imaging_type=self.run_info.imaging_type,
        )
        self._analysis_tags.xvars_shuffled = False
        self.roi = copy.deepcopy(getattr(first, 'roi', None)) if self._has_images else None

        self._refresh_repeat_statistics()

    def _require_flat(self, method_name):
        if getattr(self, '_stacked', False):
            raise NotImplementedError(
                f"AtomdataVault.{method_name} is not available on a vault "
                "stacked from multi-axis runs; work on "
                "vault.atomdata(run_id) or rebuild the vault instead."
            )

    def _warn_param_mismatches(self, chunks):
        """Emit a single warning summarizing fixed-param disagreements
        across chunks (excluding the scanned xvar itself) and record the
        per-run values in ``self.param_disagreements`` for ``param_report``."""
        first = chunks[0]
        first_params = vars(first.params)
        scanned = {_decode_xvarname(name) for name in first.xvarnames}

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
            if key.startswith('_') or key in scanned:
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
                    # Collapse identical per-shot time axes back into a shared
                    # zero-copy broadcast view (concatenation materializes the
                    # per-chunk views, so re-deduplicate afterwards).
                    t_cat = _collapse_shared_time_axes(np.concatenate(t_parts, axis=0))
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
                        if t.ndim >= 2 and t.strides[0] == 0:
                            # Shared (broadcast) time axis: reordering
                            # identical rows is a no-op — keep the view.
                            trace.t = np.broadcast_to(t[0], (n_new,) + t.shape[1:])
                        else:
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
                        if t.ndim >= 2 and t.strides[0] == 0:
                            # Shared (broadcast) time axis: reordering and
                            # reshaping identical rows only changes the leading
                            # shape — keep the zero-copy view.
                            trace.t = np.broadcast_to(
                                t[0], tuple(new_shape) + t.shape[1:]
                            )
                        else:
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
        self._require_flat('set_xvar')
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

        # The frames (and od_raw, if it was materialized) were restructured
        # above; analyze_ods recomputes od from whichever is present. od_raw
        # is on-demand now, so its presence is no longer the signal that
        # images exist.
        if reanalyze and getattr(self, '_has_images', True):
            self.analyze_ods()
        elif not getattr(self, '_has_images', True):
            self._clear_image_analysis_attrs()
        if refresh_statistics:
            self._refresh_repeat_statistics()
        return self

    # ------------------------------------------------------------------
    # Relabelling an xvar axis (values change, the data layout does not)
    # ------------------------------------------------------------------
    def _resolve_xvar_idx(self, xvar):
        """Axis index from an int (negative allowed) or an xvar name."""
        names = [_decode_xvarname(n) for n in self.xvarnames]
        if isinstance(xvar, (int, np.integer)) and not isinstance(xvar, bool):
            idx = int(xvar)
            if not -len(names) <= idx < len(names):
                raise IndexError(f"xvar index {idx} is out of range for xvars {names}.")
            return idx % len(names)
        key = _decode_xvarname(xvar)
        if key not in names:
            raise KeyError(f"{key!r} is not an xvar of this vault; xvars are {names}.")
        return names.index(key)

    def _swap_xvar(self, idx, new_key, values, *, overwrite=False):
        """Install ``values`` as xvar ``idx`` under ``new_key`` and as a param.

        Only the axis label changes: no array is reordered, so ``values[i]``
        must describe the same shots ``xvars[idx][i]`` did. The old xvar stays
        in ``params`` under its own key.
        """
        new_key = str(new_key)
        names = [_decode_xvarname(n) for n in self.xvarnames]
        old_key = names[idx]
        if new_key in names[:idx] + names[idx + 1:]:
            raise ValueError(f"{new_key!r} is already the xvar on another axis ({names}).")

        values = np.array(values, copy=True)
        n = int(self.xvardims[idx])
        if values.shape != (n,):
            raise ValueError(
                f"New values for xvar {old_key!r} (axis {idx}) must have shape ({n},); "
                f"got {values.shape}.")
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise ValueError(
                f"New values for xvar {old_key!r} contain NaN or inf; an xvar must be "
                "finite for grouping and repeat statistics.")

        existing = vars(self.params).get(new_key, None)
        if existing is not None and new_key != old_key and not overwrite:
            same = (np.shape(existing) == values.shape
                    and np.array_equal(np.asarray(existing), values))
            if not same:
                raise ValueError(
                    f"Param {new_key!r} already exists with different values; pass "
                    "overwrite=True to replace it, or choose another key.")

        n_unique_old = np.unique(np.asarray(self.xvars[idx])).size
        if np.unique(values).size != n_unique_old:
            warnings.warn(
                f"Remapping xvar {old_key!r} -> {new_key!r} changed the number of distinct "
                f"values ({n_unique_old} -> {np.unique(values).size}); repeat grouping on "
                "this axis will change.")

        names[idx] = new_key
        self.xvarnames = names
        self.xvars[idx] = values
        setattr(self.params, new_key, values)
        if not hasattr(self, 'xvar_remaps'):
            self.xvar_remaps = []
        self.xvar_remaps.append((idx, old_key, new_key))
        self._refresh_repeat_statistics()
        return self

    def remap_xvar(self, xvar, func=None, new_key=None, *, overwrite=False):
        """Replace one xvar axis by a function of it, or by an existing param.

        Parameters
        ----------
        xvar : int or str
            Axis index, or the xvar's name.
        func : callable, optional
            Maps the old xvar values to the new ones, e.g.
            ``lambda v: P0 * v / 0.444``. Called once with the whole array; if
            that fails or does not return one value per element it is applied
            element by element. Omit it to swap in an existing param.
        new_key : str
            Name of the new xvar. With ``func`` it is the name the generated
            values are stored under in ``params``; without ``func`` it must
            name an existing param whose shape already matches the axis
            (otherwise ``ValueError``). ``remap_xvar(xvar, 'key')`` is accepted
            as shorthand for the latter.
        overwrite : bool
            Allow ``func`` output to replace an existing, different param.

        The new values become ``xvars[i]`` and ``params.<new_key>``, the name
        replaces ``xvarnames[i]``, and the old xvar stays in ``params``. Shots
        are not reordered, so a non-monotonic map leaves the axis unsorted.
        Returns ``self``.
        """
        if isinstance(func, str) and new_key is None:
            func, new_key = None, func
        if new_key is None:
            raise TypeError("remap_xvar needs new_key: the name of the new xvar.")
        idx = self._resolve_xvar_idx(xvar)
        old = np.asarray(self.xvars[idx])

        if func is None:
            if new_key not in vars(self.params):
                raise KeyError(
                    f"No func was given, so {new_key!r} must be an existing param; it is not.")
            values = np.asarray(vars(self.params)[new_key])
            if values.shape != old.shape:
                raise ValueError(
                    f"Param {new_key!r} has shape {values.shape} but xvar "
                    f"{_decode_xvarname(self.xvarnames[idx])!r} has shape {old.shape}; pass a "
                    "func to generate the new values instead.")
            return self._swap_xvar(idx, new_key, values, overwrite=True)

        if not callable(func):
            raise TypeError(f"func must be callable or None; got {type(func).__name__}.")
        try:
            values = np.asarray(func(old))
            if values.shape != old.shape:
                raise ValueError
        except (TypeError, ValueError):
            values = np.array([func(v) for v in old])
        return self._swap_xvar(idx, new_key, values, overwrite=overwrite)

    def data_container_to_xvar(self, xvar, data_key, func=None, new_key=None, *,
                               reduce=None, overwrite=False):
        """Replace one xvar axis by a recorded data container (``vault.data.<key>``).

        Parameters
        ----------
        xvar : int or str
            Axis index, or the xvar's name.
        data_key : str
            Key into ``vault.data``. The container must be either 1-D with one
            value per point of the axis, or scan-shaped (``xvardims``), in which
            case it has to collapse onto that axis.
        func : callable, optional
            Applied to the collapsed values (see :meth:`remap_xvar`).
        new_key : str, optional
            Name of the new xvar and param; defaults to ``data_key``.
        reduce : {None, 'mean', 'median', 'first'}
            How a scan-shaped container collapses over the other axes. ``None``
            (default) requires it to be constant along them and raises
            otherwise; the reducers ignore NaN cells (``stack_mask`` holes).

        A per-shot record differs between repeats, so using one on the axis
        that carries the repeats removes the repeat grouping. Returns ``self``.
        """
        idx = self._resolve_xvar_idx(xvar)
        data_key = str(data_key)
        if data_key not in list(self.data.keys):
            raise KeyError(
                f"{data_key!r} is not a data container; available: {list(self.data.keys)}.")
        arr = np.asarray(getattr(self.data, data_key), dtype=float)
        n = int(self.xvardims[idx])
        dims = tuple(int(d) for d in self.xvardims)

        if arr.shape == (n,):
            values = arr
        elif arr.shape == dims:
            other = tuple(a for a in range(len(dims)) if a != idx)
            reducers = {'mean': np.nanmean, 'median': np.nanmedian}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN slices
                if reduce is None:
                    lo, hi = np.nanmin(arr, axis=other), np.nanmax(arr, axis=other)
                    if not np.allclose(lo, hi, equal_nan=True):
                        raise ValueError(
                            f"Data container {data_key!r} varies along the other xvar axes "
                            f"(spread up to {np.nanmax(hi - lo):.4g}); pass reduce='mean', "
                            "'median' or 'first' to collapse it.")
                    values = lo
                elif reduce in reducers:
                    values = reducers[reduce](arr, axis=other)
                elif reduce == 'first':
                    moved = np.moveaxis(arr, idx, 0).reshape(n, -1)
                    values = np.array([row[np.isfinite(row)][0] if np.isfinite(row).any()
                                       else np.nan for row in moved])
                else:
                    raise ValueError(
                        f"reduce must be None, 'mean', 'median' or 'first'; got {reduce!r}.")
        else:
            raise ValueError(
                f"Data container {data_key!r} has shape {arr.shape}; expected ({n},) or the "
                f"scan shape {dims}. Per-shot arrays cannot label an axis.")

        if func is not None:
            if not callable(func):
                raise TypeError(f"func must be callable or None; got {type(func).__name__}.")
            try:
                mapped = np.asarray(func(values))
                if mapped.shape != values.shape:
                    raise ValueError
            except (TypeError, ValueError):
                mapped = np.array([func(v) for v in values])
            values = mapped
        return self._swap_xvar(idx, data_key if new_key is None else new_key, values,
                               overwrite=overwrite)

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
        self._require_flat('flatten_xvar')
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

            if reanalyze and getattr(self, '_has_images', True):
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

        if reanalyze and getattr(self, '_has_images', True):
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
        inverse = np.asarray(inverse)
        # The vault sorts its shots by xvar (sort=True, the default), so the
        # groups are contiguous runs of labels 0..n_groups-1 in order. Then
        # np.add.reduceat does the grouped sum in one pass, ~3.5x faster than
        # np.add.at (measured on the arrays a vault actually reduces).
        # np.add.at stays as the general path for unsorted shots.
        contiguous = (
            inverse.size == arr.shape[0]
            and inverse.size > 0
            and np.all(np.diff(inverse) >= 0)
            and inverse[0] == 0
            and inverse[-1] == n_groups - 1
        )
        if contiguous:
            starts = np.searchsorted(inverse, np.arange(n_groups))
            csum = np.add.reduceat(vals, starts, axis=0)
            sqsum = np.add.reduceat(vals * vals, starts, axis=0)
            ncnt = np.add.reduceat(finite.astype(np.float64), starts, axis=0)
        else:
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

    def _copy_metadata_to_ragged_sibling(self, ad_out, unique_xvar, kind=None):
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

        ad_out.data = _VaultDataVault(self, kind)
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

        # If every unique xvar value ended up with the same repeat count
        # (e.g. non-overlapping merged ranges from equal-N_repeats runs, or
        # overlapping ranges that happen to sum to a uniform count), reflect
        # that in N_repeats so code reading it directly (rather than through
        # the grouped stats) sees the true per-point repeat count.
        if counts.size and np.all(counts == counts[0]):
            self.params.N_repeats = int(counts[0])

        ad_avg = object.__new__(self.__class__)
        ad_std = object.__new__(self.__class__)
        ad_sem = object.__new__(self.__class__)
        for sib, kind in ((ad_avg, 'mean'), (ad_std, 'std'), (ad_sem, 'sem')):
            self._copy_metadata_to_ragged_sibling(sib, unique, kind)

        skip = self._stat_skip_keys()

        def _reduce(value):
            mean, std = self._grouped_mean_std(value, inverse, n_groups, counts)
            sem = self._sem_from_std(std, counts)
            return mean, std, sem

        # Top-level scan-shaped arrays (od, atom_number, fits, ...). Small
        # ones are reduced now; the raw frames, a materialized od_raw and the
        # like are reduced on first access (REPEAT_STAT_LAZY_BYTES), exactly
        # as the base class does -- they cost seconds each and are rarely
        # read through the siblings.
        lazy_attrs = set()
        for key, value in vars(self).items():
            if key in skip or key.startswith('_'):
                continue
            if self._is_scan_shaped_numeric_array(value):
                if value.nbytes >= REPEAT_STAT_LAZY_BYTES:
                    lazy_attrs.add(key)
                    continue
                mean, std, sem = _reduce(value)
                vars(ad_avg)[key] = mean
                vars(ad_std)[key] = std
                vars(ad_sem)[key] = sem
            else:
                for sib in (ad_avg, ad_std, ad_sem):
                    if key not in vars(sib):
                        vars(sib)[key] = value

        # od_raw is computed on demand; the siblings reduce it on demand too.
        if getattr(self, '_has_images', True) and 'od_raw' not in vars(ad_avg):
            lazy_attrs.add('od_raw')

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

        # Scope data: large, so reduced on first access.
        if hasattr(self, 'scope_data'):
            if self.scope_data:
                lazy_attrs.add('scope_data')
            else:
                for sib in (ad_avg, ad_std, ad_sem):
                    sib.scope_data = {}

        shared = {}

        def _resolve(name, kind):
            if name not in shared:
                if name == 'scope_data':
                    shared[name] = self._reduce_scope_data_grouped(_reduce)
                else:
                    shared[name] = _reduce(getattr(self, name))
            return shared[name][{'mean': 0, 'std': 1, 'sem': 2}[kind]]

        self._install_lazy_stats(
            ((ad_avg, 'mean'), (ad_std, 'std'), (ad_sem, 'sem')),
            lazy_attrs, _resolve,
        )

        self.avg = ad_avg
        self.std = ad_std
        self.sem = ad_sem
        self._repeat_lazy_stat_context = None

    def _reduce_scope_data_grouped(self, reduce):
        """(avg, std, sem) scope_data dicts, grouped by unique xvar value.

        Best effort: on failure a warning is emitted and empty dicts are
        returned, so the siblings still have a ``scope_data``.
        """
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
                            mean, std, sem = reduce(val)
                        else:
                            mean = std = sem = val
                        out[ax] = (mean, std, sem)
                    avg_scope[scope_key][ch] = ScopeTraceArray(
                        scope_key, ch, out['t'][0], out['v'][0])
                    std_scope[scope_key][ch] = ScopeTraceArray(
                        scope_key, ch, out['t'][1], out['v'][1])
                    sem_scope[scope_key][ch] = ScopeTraceArray(
                        scope_key, ch, out['t'][2], out['v'][2])
        except Exception as e:
            warnings.warn(
                f"AtomdataVault: failed to reduce scope_data statistics "
                f"({e}); scope stats unavailable.",
                stacklevel=2,
            )
            return {}, {}, {}
        return avg_scope, std_scope, sem_scope

    def _refresh_repeat_statistics(self):
        """Override: group by unique xvar value so overlapping ranges with
        ragged repeat counts average correctly."""
        if not self._merge_overlap or int(getattr(self, 'Nvars', 1)) != 1:
            return atomdata_base._refresh_repeat_statistics(self)
        self._build_grouped_statistics()

    def _grouped_array_stats(self, arr):
        """(mean, std, sem) of a scan-shaped array grouped by unique xvar
        value. SEM uses each group's own count, so ragged repeat counts from
        overlapping merged ranges are handled correctly."""
        xvar = np.asarray(self.xvars[0])
        unique, inverse, counts = np.unique(
            xvar, return_inverse=True, return_counts=True
        )
        inverse = np.asarray(inverse).ravel()
        mean, std = self._grouped_mean_std(arr, inverse, unique.size, counts)
        return mean, std, self._sem_from_std(std, counts)

    def _use_grouped_array_stats(self):
        """Same guard _refresh_repeat_statistics uses: the grouped path only
        applies to single-xvar vaults built with merge_overlap."""
        return self._merge_overlap and int(getattr(self, 'Nvars', 1)) == 1

    def avg_array(self, arr, xvar_idx=None, return_std=True, return_sem=True):
        """Override: groups by unique xvar value rather than a fixed repeat
        stride. See atomdata_base.avg_array."""
        if not self._use_grouped_array_stats():
            return atomdata_base.avg_array(
                self, arr, xvar_idx=xvar_idx,
                return_std=return_std, return_sem=return_sem,
            )
        arr = self._check_scan_shaped(arr, 'avg_array')
        avg, std, sem = self._grouped_array_stats(arr)
        return self._pack_array_stats(avg, std, sem, return_std, return_sem)

    def std_array(self, arr, xvar_idx=None):
        """Override: see avg_array and atomdata_base.std_array."""
        if not self._use_grouped_array_stats():
            return atomdata_base.std_array(self, arr, xvar_idx=xvar_idx)
        arr = self._check_scan_shaped(arr, 'std_array')
        return self._grouped_array_stats(arr)[1]

    def sem_array(self, arr, xvar_idx=None):
        """Override: divides each point's std by sqrt of that point's own
        repeat count. See avg_array and atomdata_base.sem_array."""
        if not self._use_grouped_array_stats():
            return atomdata_base.sem_array(self, arr, xvar_idx=xvar_idx)
        arr = self._check_scan_shaped(arr, 'sem_array')
        return self._grouped_array_stats(arr)[2]

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
        self._require_flat('collapse_to_unique')
        if getattr(self._analysis_tags, 'averaged', False):
            print('AtomdataVault is already collapsed to unique xvar values.')
            return self

        xvar = np.asarray(self.xvars[0])
        unique, inverse, counts = np.unique(
            xvar, return_inverse=True, return_counts=True
        )
        inverse = np.asarray(inverse).ravel()
        n_groups = unique.size

        # Collapsing averages the ODs, not the frames: the OD of a mean frame
        # is not the mean OD. Materialize the full-frame OD first so it is
        # averaged below and analyze_ods crops that, as it always did.
        if getattr(self, '_has_images', True):
            _ = self.od_raw

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

        if reanalyze and getattr(self, '_has_images', True):
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
        self._require_flat('drop_runs')
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

        if reanalyze and getattr(self, '_has_images', True):
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
                       skip_missing=True, roi_id=None, lite=True,
                       uniform_roi=True, **kwargs):
        """Build a vault from a contiguous run-id range ``[start_id, stop_id]``.

        Missing/aborted run-ids are skipped (``skip_missing``). If
        ``experiment_name`` is given, only runs whose experiment class or
        filepath contains that substring are kept -- handy for an experiment
        builder that interleaves several experiment types. When ``uniform_roi``
        is True (default), subsequent runs reuse the first loaded run's ROI so
        the selector opens at most once.
        """
        start_id, stop_id = int(start_id), int(stop_id)
        if stop_id < start_id:
            start_id, stop_id = stop_id, start_id

        ads, skipped = [], []
        anchor_roi = roi_id
        for rid in range(start_id, stop_id + 1):
            # With uniform_roi, reuse the first loaded run's ROI for every
            # subsequent load; otherwise honor the caller's roi_id per run.
            load_roi_id = anchor_roi if uniform_roi else roi_id
            try:
                ad = atomdata(rid, roi_id=load_roi_id, lite=lite)
            except Exception:
                if skip_missing:
                    skipped.append(rid)
                    continue
                raise
            if uniform_roi and anchor_roi is None:
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
        return cls(ads, roi_id=roi_id, lite=lite, uniform_roi=uniform_roi,
                   **kwargs)

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
        self._require_flat('recrop')
        if not getattr(self, '_has_images', True):
            print("no images in dataset (ignore_images), no roi to crop")
            return
        if self._lite:
            raise NotImplementedError(
                "recrop is not supported on a lite AtomdataVault; the lite "
                "runs are already cropped. Build the vault from full "
                "(non-lite) runs to recrop all runs."
            )
        # Selecting one ROI here and re-running analyze_ods re-crops every
        # run together, since the frames of all runs share one leading axis.
        # The GUI is given the full-frame ODs only if they already exist;
        # otherwise it computes the OD of each frame it displays from the
        # first run's raw images (self.roi._images), which is far cheaper
        # than materializing od_raw for the whole vault.
        od_raw = vars(self).get('od_raw')
        od_flat = (od_raw.reshape(-1, *od_raw.shape[-2:])
                   if od_raw is not None else None)
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
