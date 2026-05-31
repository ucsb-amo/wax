"""AtomdataVault: concatenate several single-axis atomdata runs into one
analysis object.

Intended use case: a 1-D scan was broken across multiple runs to keep per-file
sizes manageable. AtomdataVault stitches the chunks back together so the
combined dataset can be analyzed with the usual atomdata interface
(``vault.atom_number``, ``vault.od``, ``vault.fit_sd_x``, ``vault.data.*``,
etc.) without ever re-saving anything to disk.

Limitations (v1):
    * Only 1-D scans are supported. Every input must have ``Nvars == 1``.
    * All inputs must share the same ``xvarnames[0]`` unless
      ``xvarname_override=True``.
    * All inputs must share ``N_repeats``, ``imaging_type``, and per-shot
      image shape.
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
    """Flatten a scalar / list / tuple / ndarray of mixed types into a list."""
    if inputs is None:
        raise ValueError("AtomdataVault requires at least one input.")
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
    """

    def __init__(self,
                 inputs,
                 roi_id=None,
                 lite=False,
                 xvarname_override=False,
                 sort=True):

        # Lightweight book-keeping expected by inherited helpers.
        self._lite = lite
        self._timing_enabled = False
        self._timing = {}
        self.server_talk = None

        self.avg: Optional[atomdata_base] = None
        self.std: Optional[atomdata_base] = None
        self.sem: Optional[atomdata_base] = None
        self._repeat_sem_source = None
        self._repeat_sem_divisor = None
        self._repeat_zero_proxy = None
        self._repeat_lazy_stat_context = None
        self._data_file_path = None
        self._saved_roi_from_file = False

        # 1. Normalize and materialize inputs.
        raw_inputs = _flatten_inputs(inputs)
        if len(raw_inputs) == 0:
            raise ValueError("AtomdataVault requires at least one input.")

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
                    if _has_subsequent_int_loads or lite:
                        item.save_roi_h5()
            elif isinstance(item, (int, np.integer)):
                # First int load: use caller-supplied roi_id (may open GUI once).
                # Subsequent int loads: reuse the first run's roi_id so no
                # additional GUI opens.
                if _first_run_id is None:
                    ad = atomdata(int(item), roi_id=roi_id, lite=lite)
                    _first_run_id = int(ad.run_info.run_id)
                    # Persist the ROI so subsequent int loads (and lite-dataset
                    # creation for each run) can find it by run_id.
                    if _has_subsequent_int_loads or lite:
                        ad.save_roi_h5()
                else:
                    _roi = roi_id if roi_id is not None else _first_run_id
                    ad = atomdata(int(item), roi_id=_roi, lite=lite)
                ads.append(ad)
            else:
                raise TypeError(
                    f"AtomdataVault inputs must be atomdata objects or "
                    f"run_id ints, got {type(item).__name__}."
                )

        # 2. Validate compatibility.
        self._validate_inputs(ads, xvarname_override)

        # 3. Deep-copy and unshuffle each chunk so the caller's handles are
        #    untouched and the per-shot arrays are in xvar order on axis 0.
        chunks = []
        for ad in ads:
            ad_copy = copy.deepcopy(ad)
            if getattr(ad_copy._analysis_tags, 'xvars_shuffled', False):
                ad_copy.unshuffle(reanalyze=False)
                if getattr(ad_copy, '_has_images', True):
                    ad_copy._sort_images()
            chunks.append(ad_copy)

        # 4. Assemble vault state from chunks.
        first = chunks[0]
        xvarname = _decode_xvarname(first.xvarnames[0])
        self.source_run_ids = [int(c.run_info.run_id) for c in chunks]

        # params: deep-copy first, then patch the scanned attribute below.
        self.params = copy.deepcopy(first.params)
        self.p = self.params
        self.camera_params = copy.deepcopy(first.camera_params)
        self.run_info = copy.deepcopy(first.run_info)
        self.experiment_code = getattr(first, 'experiment_code', None)
        self._has_images = bool(getattr(first, '_has_images', True))

        self._warn_param_mismatches(chunks)

        # Concatenate xvar values (axis 0).
        xvar_values = np.concatenate(
            [np.asarray(c.xvars[0]) for c in chunks], axis=0
        )

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
        self._maybe_concat_scope_data(chunks)

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

        # ROI: pass only the first chunk's images for display/selection so the
        # GUI shows a representative frame from the first run, not all runs.
        # The resulting roix/roiy coordinates are then applied to the full
        # concatenated od_raw during analyze_ods.
        if self._has_images:
            roi_source = roi_id if roi_id is not None else int(first.run_info.run_id)
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, ads, xvarname_override):
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
            if first_has_images else None
        )

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
            if int(ad.params.N_repeats) != first_nrep:
                raise ValueError(
                    f"N_repeats mismatch: run {ad.run_info.run_id} has "
                    f"N_repeats={ad.params.N_repeats} but first input has "
                    f"N_repeats={first_nrep}."
                )
            if ad.run_info.imaging_type != first_imgtype:
                raise ValueError(
                    f"imaging_type mismatch on run {ad.run_info.run_id}."
                )
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

    def _warn_param_mismatches(self, chunks):
        """Emit a single warning summarizing fixed-param disagreements
        across chunks (excluding the scanned xvar itself)."""
        first = chunks[0]
        first_params = vars(first.params)
        xvarname = _decode_xvarname(first.xvarnames[0])

        mismatched = []
        for key, first_val in first_params.items():
            if key.startswith('_') or key == xvarname:
                continue
            for c in chunks[1:]:
                other = vars(c.params).get(key, None)
                if other is None:
                    continue
                try:
                    if isinstance(first_val, np.ndarray) or isinstance(other, np.ndarray):
                        a = np.asarray(first_val)
                        b = np.asarray(other)
                        if a.shape != b.shape or not np.array_equal(a, b):
                            mismatched.append(key)
                            break
                    else:
                        if first_val != other:
                            mismatched.append(key)
                            break
                except Exception:
                    # Non-comparable params are silently skipped.
                    continue

        if mismatched:
            warnings.warn(
                "AtomdataVault: fixed parameters disagree across input runs "
                "(using values from the first run): "
                + ", ".join(sorted(set(mismatched))),
                stacklevel=2,
            )

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

    def _maybe_concat_scope_data(self, chunks):
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

        xvar_values = xvar_values[order]
        self.xvars[0] = xvar_values
        setattr(self.params, self.xvarnames[0], xvar_values)

        if self._has_images:
            Nf = int(self.params.N_pwa_per_shot) + 2
            n_shots = len(order)
            if self.images.shape[0] == n_shots * Nf:
                # Raw interleaved: (N_shots * Nf, H, W). Sort in groups of Nf
                # so that each shot's (atoms, light, dark) frames stay together.
                frame_order = np.concatenate(
                    [np.arange(i * Nf, (i + 1) * Nf) for i in order]
                )
                self.images = self.images[frame_order]
                self.image_timestamps = self.image_timestamps[frame_order]
            else:
                # Already 1 image per shot (lite or special mode).
                self.images = self.images[order]
                self.image_timestamps = self.image_timestamps[order]

        for k in self.data.keys:
            arr = vars(self.data)[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == len(order):
                vars(self.data)[k] = arr[order]

        if hasattr(self, 'scope_data'):
            for scope_key, ch_dict in self.scope_data.items():
                for ch, trace in ch_dict.items():
                    t = np.asarray(trace.t)
                    v = np.asarray(trace.v)
                    if t.ndim >= 1 and t.shape[0] == len(order):
                        trace.t = t[order]
                    if v.ndim >= 1 and v.shape[0] == len(order):
                        trace.v = v[order]

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
