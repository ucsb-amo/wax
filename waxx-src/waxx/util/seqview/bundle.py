"""The seqview bundle -- what a pulse-sequence viewer window is fed.

A bundle is machine-agnostic: lanes of samples or edges, pulses with their
provenance (source line, macro, parameter label/value, requested vs actual
length), point events (phase resets, frequency changes), shot bands,
framing bands, source texts, a per-shot parameter table and warnings. All
times are **nanoseconds** (float64) from the start of the simulated
timeline. The producer (kexp.control.opx.viewer for the OPX) does every
lab-specific step -- element names, polarities, unit formatting -- so this
module and the window need only numpy.

On disk a bundle is one ``.npz``: the JSON metadata under the key ``meta``
and each sample/edge array under the key the metadata names. Written to a
temporary name and renamed into place, so a viewer process never sees a
half-written file.

Schema (version 1) -- keys of ``meta``:

    title, subtitle          str
    t_end_ns                 float   end of the simulated window
    dt_ns                    float   analog sample period (1.0 for the OPX+)
    lanes                    [lane]  display order
    pulses                   [pulse]
    events                   [event]
    shots                    [shot]
    framing                  [band]
    sources                  {id: source}
    params                   {names: [str], columns: {name: [float per shot]},
                              display: {name: [str per shot]}}
    warnings                 [{level: 'error'|'warning'|'info', text: str}]
    info                     dict (free-form provenance shown in the About tab)

    lane   {id, label, kind: 'semantic'|'digital'|'analog'|'adc',
            group: 'semantic'|'physical', color: '#rrggbb', element, port,
            role, note, high_label, low_label,
            samples: npz key (analog) | null,
            edges: npz key (digital: sorted edge times ns) | null,
            level0: int (digital level before the first edge),
            pulses: [pulse ids], events: [event ids],
            visible: bool, height: float (relative)}
    pulse  {id, lane, element, op, pulse_name, t0, t1, shot, phase:
            'prologue'|'handshake'|'body'|'epilogue', macro, label, value,
            value_str, requested_ns, actual_ns, rounded: bool,
            src_line, src_lines, qua_line, note, data_key, kind:
            'play'|'exposure'|'adc'|'wait'|'trigger'|'latch', color,
            text (short label drawn on the bar), truncated: bool}
    event  {id, lane, t, kind, label, detail, src_line, qua_line,
            approx: bool, shot}
    shot   {index, t0, t1, label, params: {name: value_str}}
    band   {t0, t1, shot, kind: 'handoff'|'handback'|'not_simulated',
            label}
    source {title, path, text, first_line, language}
"""

import json
import os
import tempfile

import numpy as np

VERSION = 1

LANE_KINDS = ('semantic', 'digital', 'analog', 'adc')
PULSE_KINDS = ('play', 'exposure', 'adc', 'wait', 'trigger', 'latch')
PHASES = ('prologue', 'handshake', 'body', 'epilogue')


def empty_meta(title='pulse sequence'):
    return {
        'version': VERSION, 'title': title, 'subtitle': '',
        't_end_ns': 0.0, 'dt_ns': 1.0,
        'lanes': [], 'pulses': [], 'events': [], 'shots': [], 'framing': [],
        'sources': {}, 'params': {'names': [], 'columns': {}, 'display': {}},
        'warnings': [], 'info': {},
    }


class Bundle:
    """meta (dict) + arrays ({npz key: ndarray})."""

    def __init__(self, meta=None, arrays=None):
        self.meta = meta if meta is not None else empty_meta()
        self.arrays = dict(arrays or {})

    # ---- convenience -------------------------------------------------
    @property
    def lanes(self):
        return self.meta['lanes']

    @property
    def pulses(self):
        return self.meta['pulses']

    @property
    def events(self):
        return self.meta['events']

    def lane(self, lane_id):
        for ln in self.meta['lanes']:
            if ln['id'] == lane_id:
                return ln
        raise KeyError(lane_id)

    def array(self, key):
        return self.arrays[key]

    def validate(self):
        """Raise ValueError on a structurally broken bundle (missing arrays,
        dangling ids, unsorted edges)."""
        m = self.meta
        if int(m.get('version', 0)) != VERSION:
            raise ValueError(f"seqview bundle version {m.get('version')!r} "
                             f"is not {VERSION}")
        lane_ids = {ln['id'] for ln in m['lanes']}
        if len(lane_ids) != len(m['lanes']):
            raise ValueError("duplicate lane ids")
        for ln in m['lanes']:
            if ln['kind'] not in LANE_KINDS:
                raise ValueError(f"lane {ln['id']!r}: kind {ln['kind']!r}")
            for key in ('samples', 'edges'):
                k = ln.get(key)
                if k is not None and k not in self.arrays:
                    raise ValueError(f"lane {ln['id']!r}: missing array {k!r}")
            if ln.get('edges') is not None:
                e = self.arrays[ln['edges']]
                if e.ndim != 1 or (e.size > 1 and np.any(np.diff(e) < 0)):
                    raise ValueError(f"lane {ln['id']!r}: edges not sorted")
        for p in m['pulses']:
            if p['lane'] not in lane_ids:
                raise ValueError(f"pulse {p['id']}: lane {p['lane']!r}")
            if p['kind'] not in PULSE_KINDS:
                raise ValueError(f"pulse {p['id']}: kind {p['kind']!r}")
            if not (p['t1'] >= p['t0']):
                raise ValueError(f"pulse {p['id']}: t1 < t0")
        for ev in m['events']:
            if ev['lane'] not in lane_ids:
                raise ValueError(f"event {ev['id']}: lane {ev['lane']!r}")
        return self

    # ---- I/O ---------------------------------------------------------
    def save(self, path):
        path = os.fspath(path)
        d = os.path.dirname(path) or '.'
        fd, tmp = tempfile.mkstemp(prefix='.seqview-', suffix='.npz', dir=d)
        os.close(fd)
        try:
            payload = {k: np.asarray(v) for k, v in self.arrays.items()}
            payload['meta'] = np.array(json.dumps(self.meta, default=_json_default))
            with open(tmp, 'wb') as f:
                np.savez(f, **payload)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path):
        with np.load(os.fspath(path), allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            arrays = {k: z[k] for k in z.files if k != 'meta'}
        return cls(meta, arrays).validate()

    @classmethod
    def coerce(cls, obj):
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, (str, os.PathLike)):
            return cls.load(obj)
        if isinstance(obj, dict) and 'meta' in obj:
            return cls(obj['meta'], obj.get('arrays', {}))
        raise TypeError(f"cannot make a seqview Bundle from {type(obj)!r}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def edges_from_samples(samples, level0=None):
    """Run-length encode a 0/1 sample array (1 sample per dt) into
    (level0, edge_indices) -- the index of every sample whose value differs
    from the previous one."""
    s = np.asarray(samples).astype(np.int8).ravel()
    if s.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    idx = np.flatnonzero(np.diff(s)) + 1
    return int(s[0]), idx.astype(np.int64)


def level_at(edges, level0, t):
    """Digital level at time t given sorted edge times and the initial
    level (each edge toggles)."""
    n = int(np.searchsorted(np.asarray(edges), t, side='right'))
    return int(level0) ^ (n & 1)
