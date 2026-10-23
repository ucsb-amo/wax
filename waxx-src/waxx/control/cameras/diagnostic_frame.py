"""Per-shot diagnostic frames: a frame of named images, each taken by a
camera every shot, and the lookup of a camera's trigger TTL on an
experiment's ttl frame.

A lab's ``camera_id`` subclasses ``diagnostic_frame`` and assigns its
images between ``setup()`` and ``cleanup()``, like the device frames::

    class diagnostic_frame(diagnostic_frame_waxx):
        def __init__(self):
            self.setup()
            self.img_mot = self.image(cameras.mot_basler, t_offset=-150e-3, t_stage='t_mot_load',
                                      exposure_time=300e-6, gain=3., roi=[...], show='image')
            self.cleanup()

An image is one frame a camera -- an entry of the lab's camera frame,
``cameras.<key>`` -- takes every shot, at its own ``exposure_time`` (s),
``gain`` (dB) and ``roi`` ``[x1, y1, x2, y2]`` (sensor pixels, end-exclusive).
What an image does not give comes from its camera entry: the exposure and
gain the entry holds for the frame's ``default_imaging_type`` (absorption
unless the lab's ``setup()`` sets another), and the full frame. The cameras
are the entries the images name, each known by its key; one stream per
camera serves all its images.

An image's key -- the DataVault key it is saved under -- is the attribute
name. Its frame index is its place among its camera's images in the order
they are assigned: assign them in the order the sequence takes them. The
sequence takes it where it belongs with
``self.diag.img_mot.trigger()``: ``t_offset`` (s) from the timeline cursor
(negative: before it); the cursor stays where it was. Nothing is taken with
diagnostics off, during warm-up shots (``active`` cleared), or for a camera
that could not be used this run (a placeholder stream).

``t_stage`` (optional) is the length of the stage the frame sits in, in s or
as the name of an ExptParams attribute: a frame with ``|t_offset| > t_stage`` would
fall outside its stage -- before it starts, possibly in the past -- and is
skipped every shot, which is said at prepare. It is resolved from the run's
params when the run is bound and again at finish_prepare (``resolve_stages``);
a scanned param counts by its shortest value.

For a run, the lab builds a fresh frame and binds it (``bind``) with its own
stream factory and TTL lookup. Each image then holds its camera's trigger TTL
and reports its edge to its camera's stream from the host, so every image is
the same kind of object to the ARTIQ compiler and no per-camera code is made.
"""

import copy

import numpy as np

from artiq.experiment import kernel, rpc, delay, now_mu

from waxx.control.cameras.camera_param_classes import CameraParams, img_types
from waxx.control.cameras.camera_stream_client import (
    InactiveCameraStream, TriggeredCameraStreamClient)
from waxx.control.cameras.exposure_delay import ExposureDelayCell, dummy_cell

_SHOW = ('image', 'counts', None)
SETTING_KEYS = ('exposure_time', 'gain', 'roi')
DEFAULT_IMAGING_TYPE = img_types.ABSORPTION


def camera_trigger_ttl(ttl, camera):
    """The TTL object on the experiment's ttl frame `ttl` wired to `camera`'s
    trigger input: the frame's attribute named by the entry's `trigger_ttl`,
    or, failing that, its TTL on channel `trigger_ttl_ch`. The entry's own
    `_trigger_ttl` object is never returned -- it belongs to the camera
    config's unbound frame, not to the experiment's devices. Raises
    ValueError for an entry with no trigger, or one the frame does not have."""
    name = getattr(camera, 'trigger_ttl', '')
    ch = int(getattr(camera, 'trigger_ttl_ch', -1))
    if not name and ch < 0:
        raise ValueError(
            f"No camera TTL mapping found for camera key '{camera.key}' "
            f"(set trigger_ttl on its entry in the camera config, camera_id).")
    if name and hasattr(ttl, name):
        return getattr(ttl, name)
    if ch >= 0:
        for t in getattr(ttl, 'ttl_list', ()):
            if getattr(t, 'ch', None) == ch:
                return t
    raise ValueError(
        f"Camera '{camera.key}' names trigger_ttl '{name}' (ch {ch}), which "
        f"is not a TTL on the experiment's ttl frame (ttl_id).")


def _connected(stream):
    """True: a triggered stream that takes frames (connected, registered, not
    dead). False: no frames this run (a placeholder, diagnostics off, or dead).
    None: built without a connection and not connected yet (the claim
    connects it)."""
    if not isinstance(stream, TriggeredCameraStreamClient):
        return False
    if getattr(stream, '_dead_reason', None) is not None:
        return False
    if not getattr(stream, '_connected', True):
        return None
    return bool(getattr(stream, 'live', True))


def _exposure_delay_record(img):
    """An image's exposure delay for the run's record: its stream's
    (TriggeredCameraStreamClient.exposure_delay_record: the value the kernel
    uses, its source, how it reached the kernel, the pre-compile value, the
    camera's report at the claim), or None values for an image that takes no
    frames."""
    rec = getattr(img._stream, 'exposure_delay_record', None) if img.live else None
    if rec is None:
        return {"exposure_delay": None, "exposure_delay_source": None}
    try:
        return dict(rec())
    except Exception as e:
        return {"exposure_delay": None, "exposure_delay_source": f"unknown ({e!r})"}


def _number(v):
    return float(f"{float(v):.4g}")


def _as_roi(roi, what):
    """An ROI as ``[x1, y1, x2, y2]`` ints, None for the full frame."""
    if roi is None:
        return None
    r = [int(v) for v in roi]
    if len(r) != 4 or min(r) < 0 or r[2] <= r[0] or r[3] <= r[1]:
        raise ValueError(f"{what}: roi={list(roi)!r} is not [x1, y1, x2, y2] with "
                         f"0 <= x1 < x2 and 0 <= y1 < y2")
    return r


def camera_defaults(camera, imaging_type=DEFAULT_IMAGING_TYPE):
    """(exposure_time s, gain dB) a camera entry holds for ``imaging_type``.
    Read from a copy: the entries are shared, and the run camera's
    ``select_imaging_type`` changes its own."""
    c = copy.copy(camera)
    c.select_imaging_type(imaging_type)
    return float(c.exposure_time), float(c.gain)


class DiagnosticCamera:
    """A camera of the diagnostics: its camera entry and the images it takes,
    in frame-index order. The frame makes these from its images
    (``entries()``); one is named by its entry's key."""

    def __init__(self, camera, images=()):
        self.camera = camera
        self.images = list(images)
        self.stream = None      # set by bind

    @property
    def key(self):
        return str(self.camera.key)

    @property
    def label(self):
        return self.key

    @property
    def frames(self):
        """The frame keys, in frame-index order."""
        return [img.key for img in self.images]

    @property
    def frame_settings(self):
        """{frame key: exposure / gain} for the frames taken at other settings
        than the first frame (the stream's own)."""
        if not self.images:
            return {}
        first = self.images[0].camera_settings()
        return {img.key: img.camera_settings() for img in self.images[1:]
                if img.camera_settings() != first}

    @property
    def frame_rois(self):
        """{frame key: its ROI, None = the full frame}, every frame."""
        return {img.key: img.roi for img in self.images}

    def frame_index(self, key):
        """The index of frame ``key`` on this camera, -1 if it takes none."""
        return self.frames.index(key) if key in self.frames else -1

    def assignment(self, **values):
        """Its images' lines in a frame's ``__init__`` (``values`` replace each
        one's exposure_time / gain / roi)."""
        return "\n".join(img.assignment(**values) for img in self.images)


class DiagnosticImage:
    """One frame a camera takes every shot (module docstring): ``camera`` a
    camera-frame entry, ``t_offset`` (s) from where ``trigger()`` is called,
    ``t_stage`` (s or an ExptParams name; None: no limit), the
    ``exposure_time`` (s), ``gain`` (dB) and ``roi`` it is taken at (each
    None: the entry's for ``imaging_type``, and the full frame), and for the
    diagnostic viewer a ``label`` and ``show`` ('image', 'counts' or None:
    not shown). ``exposure_time`` / ``gain`` / ``roi`` hold the values used;
    ``given`` names the ones the line gave."""

    def __init__(self, camera, t_offset=0., t_stage=None, exposure_time=None, gain=None,
                 roi=None, label=None, show='counts', imaging_type=DEFAULT_IMAGING_TYPE):
        if not isinstance(camera, CameraParams):
            raise TypeError(f"DiagnosticImage: camera must be a camera-frame entry "
                            f"(cameras.<key>), not {camera!r}")
        if not getattr(camera, 'key', ''):
            raise ValueError(f"DiagnosticImage: camera entry {camera!r} has no key "
                             f"(is it an attribute of the camera frame?)")
        if show not in _SHOW:
            raise ValueError(f"DiagnosticImage: show={show!r}, not one of {_SHOW}")
        if not (t_stage is None or isinstance(t_stage, str)):
            t_stage = float(t_stage)
        given = dict(exposure_time=exposure_time, gain=gain, roi=roi)
        self.given = tuple(k for k in SETTING_KEYS if given[k] is not None)
        default_exposure, default_gain = camera_defaults(camera, imaging_type)
        self.camera = camera
        self.t_offset = float(t_offset)
        self.t_stage = t_stage
        self.exposure_time = float(default_exposure if exposure_time is None else exposure_time)
        self.gain = float(default_gain if gain is None else gain)
        self.roi = _as_roi(roi, f"image on cameras.{camera.key}")
        if self.exposure_time <= 0.:
            raise ValueError(f"image on cameras.{camera.key}: no exposure_time given, and the "
                             f"entry has none for imaging type {imaging_type!r}")
        self._label = label
        self.show = show
        self.key = ""               # the attribute name, set by the frame
        # kernel side, set by bind (the same types for every image)
        self.frame = None           # the run's frame: frame.active
        self.core = None
        self.ttl = None             # the camera's trigger TTL
        self.j = -1                 # frame index on the camera
        self.live = False           # the camera's stream takes frames this run
        self.in_stage = True        # |t_offset| <= t_stage (resolve_stages)
        # the camera stream's exposure-delay cell (the experiment's spare one
        # for an image that takes no frames): the kernel writes the claim's
        # value into it at its start (Scanner.load_exposure_delays)
        self.exposure_delay_cell = ExposureDelayCell(0.)
        self.t_trigger = 2.e-6
        self._stream = None
        self.t_stage_s = None       # t_stage in s, as resolved

    @property
    def label(self):
        return self._label or self.key.removeprefix('img_').replace('_', ' ')

    def camera_settings(self):
        """The camera settings this frame is taken at: exposure / gain."""
        return {'exposure_time': self.exposure_time, 'gain': self.gain}

    def settings(self):
        """Exposure, gain and ROI (None = the full frame) this frame is taken at."""
        return {**self.camera_settings(), 'roi': self.roi}

    def assignment(self, **values):
        """This image's line in a frame's ``__init__``: the settings it gave,
        and ``values`` (exposure_time / gain / roi; None: not given) in place
        of them."""
        args = [f"cameras.{self.camera.key}", f"t_offset={self.t_offset!r}"]
        if self.t_stage is not None:
            args.append(f"t_stage={self.t_stage!r}")
        vals = {k: getattr(self, k) for k in self.given}
        vals.update({k: v for k, v in values.items() if k in SETTING_KEYS and v is not None})
        for k in ('exposure_time', 'gain'):
            if k in vals:
                args.append(f"{k}={_number(vals[k])!r}")
        if 'roi' in vals:
            args.append(f"roi={[int(v) for v in vals['roi']]!r}")
        if self._label:
            args.append(f"label={self._label!r}")
        if self.show != 'counts':
            args.append(f"show={self.show!r}")
        return f"self.{self.key} = self.image(" + ", ".join(args) + ")"

    @kernel
    def trigger(self):
        """Take this frame ``t_offset`` from the cursor; the cursor stays."""
        if self.frame.active and self.live and self.in_stage:
            t_delay = self.exposure_delay_cell.value
            delay(self.t_offset)
            self.announce_edge(now_mu() - self.core.get_rtio_counter_mu())
            delay(-t_delay)
            self.ttl.pulse(self.t_trigger)
            delay(t_delay - self.t_trigger - self.t_offset)

    @rpc(flags={"async"})
    def announce_edge(self, slack_mu):
        """Async RPC target: the edge, to the camera's stream (it pairs the
        frame with the shot and tells its trigger-line siblings)."""
        try:
            self._stream.announce_trigger_mu(slack_mu, self.j)
        except Exception:
            pass


class diagnostic_frame:
    """The base of a lab's frame of diagnostic images (module docstring).
    Images are named by the attributes they are assigned to; the cameras are
    the camera entries they name."""

    enabled = False
    active = False          # kernel switch: cleared around warm-up shots

    def __init__(self):
        self.setup()
        self.cleanup()

    def setup(self):
        # an image with no roi (the full frame) is stored block-averaged this much
        self.no_roi_downsample = 4
        # the camera entry's exposure / gain for this imaging type fill what an image does not give
        self.default_imaging_type = DEFAULT_IMAGING_TYPE

    def cleanup(self):
        self._check()

    def __setattr__(self, name, value):
        if isinstance(value, DiagnosticImage):
            if hasattr(diagnostic_frame, name):
                raise ValueError(f"{name!r} is taken by diagnostic_frame.{name}")
            if value.key and value.key != name:
                raise ValueError(f"{type(self).__name__}.{name} is already .{value.key}")
            if not name.isidentifier():
                raise ValueError(f"image key {name!r} is not an identifier")
            value.key = name
        object.__setattr__(self, name, value)

    def camera(self, *a, **k):
        raise TypeError("diagnostic cameras are no longer assigned: give the camera entry "
                        "to each image, self.img_x = self.image(cameras.<key>, t_offset=..., "
                        "exposure_time=..., gain=..., roi=[...])")

    def image(self, camera, t_offset=0., t_stage=None, exposure_time=None, gain=None, roi=None,
              label=None, show='counts'):
        """An image the camera entry ``camera`` takes every shot (assign it to
        an attribute: its key). Assign a camera's images in the order they
        are taken."""
        img = DiagnosticImage(camera, t_offset=t_offset, t_stage=t_stage,
                              exposure_time=exposure_time, gain=gain, roi=roi, label=label,
                              show=show, imaging_type=getattr(self, 'default_imaging_type',
                                                              DEFAULT_IMAGING_TYPE))
        self.__dict__.setdefault('_made', []).append(img)
        return img

    def images(self):
        """{key: image}, in the order assigned."""
        return {k: v for k, v in vars(self).items() if isinstance(v, DiagnosticImage)}

    def entries(self):
        """The cameras the images name, in the order first named, each with
        its images in frame-index order."""
        imgs = tuple(self.images().values())
        cached = self.__dict__.get('_entries')
        if cached is None or cached[0] != imgs:
            cams = {}
            for img in imgs:
                cam = cams.setdefault(img.camera.key, DiagnosticCamera(img.camera))
                cam.images.append(img)
            cached = (imgs, list(cams.values()))
            self.__dict__['_entries'] = cached
        return cached[1]

    def _check(self):
        unnamed = [img for img in self.__dict__.get('_made', ()) if not img.key]
        if unnamed:
            raise ValueError(f"{len(unnamed)} image(s) not assigned to an attribute of the frame "
                             f"(on " + ", ".join(sorted({i.camera.key for i in unnamed})) + ")")
        for cam in self.entries():
            serials = {str(getattr(i.camera, 'serial_no', '')) for i in cam.images}
            if len(serials) != 1:
                raise ValueError(f"camera {cam.key!r}: its images name entries with different "
                                 f"serials {sorted(serials)}")

    def bind(self, expt, enabled, make_streams, trigger_ttl):
        """Make this frame the run's: one stream per camera with images --
        ``make_streams(expt, specs)`` (the lab's camera_streams, setting them
        up together; with ``enabled`` False none is opened) -- and each image
        given its camera's trigger TTL (``trigger_ttl(camera_entry)``, the TTL
        on the experiment's ttl frame) and frame index. Returns the frame."""
        self._check()
        self.enabled = bool(enabled)
        self.active = self.enabled
        cams = [c for c in self.entries() if c.images]
        if self.enabled:
            # the stream runs at its first frame's settings; frames at others
            # are set between edges, and each frame is cropped to its own ROI
            streams = make_streams(expt, [
                dict(camera=c.camera, key=c.frames, mode='triggered', on_error='dummy',
                     exposure_time=c.images[0].exposure_time, gain=c.images[0].gain,
                     crop_to_saved_roi=False, frame_rois=c.frame_rois,
                     frame_settings=c.frame_settings or None,
                     no_roi_downsample=self.no_roi_downsample)
                for c in cams])
        else:
            streams = [InactiveCameraStream(expt, c.frames, reason="diagnostics off")
                       for c in cams]
        for cam, stream in zip(cams, streams):
            cam.stream = stream
            live = isinstance(stream, TriggeredCameraStreamClient)
            if live:
                ttl = stream.ttl
            else:
                try:
                    ttl = trigger_ttl(cam.camera)
                except Exception as e:
                    raise ValueError(f"diagnostic camera {cam.key!r}: no trigger TTL ({e})") from e
            for j, img in enumerate(cam.images):
                img.frame = self
                img.core = expt.core
                img.ttl = ttl
                img.j = j
                img.live = live
                img._stream = stream if live else None
                # the stream's cell itself, not a copy of its value: the
                # kernel writes the claim's delay into it at its start
                img.exposure_delay_cell = (stream.exposure_delay_cell if live
                                           else dummy_cell(expt))
                if live:
                    img.t_trigger = float(stream.t_trigger)
        self.resolve_stages(getattr(expt, 'params', None), quiet=True)
        # The streams may be built without a connection (queue phase 2: the
        # run's claim connects them, after the compile). ``live`` above is
        # what the kernel sees and is decided here, before the compile; a
        # stream that then fails to connect is dead -- its edges still fire,
        # nothing is stored -- and record() says so. A frame whose streams
        # are connected already reports now; otherwise the claim calls
        # report_connected().
        if self.enabled and all(_connected(c.stream) is not None for c in cams):
            self.report_connected()
        return self

    def report_connected(self):
        """One line: which cameras take frames this run (after the streams
        were connected). Nothing with diagnostics off."""
        if not self.enabled:
            return
        cams = [c for c in self.entries() if c.images]
        up = [c.key for c in cams if _connected(c.stream)]
        down = [c.key for c in cams if c.key not in up]
        print("[diag cams] connected: " + (", ".join(up) or "none")
              + (f" | NOT connected: {', '.join(down)}" if down else ""))

    def streams(self):
        """{camera key: its stream} (bound cameras with images)."""
        return {c.key: c.stream for c in self.entries() if c.stream is not None}

    def resolve_stages(self, params, xvars=(), quiet=False):
        """Each image's ``t_stage`` in s from ``params`` (a scanned key -- in
        ``xvars`` -- by its shortest value) and whether ``|t_offset|`` fits in it.
        A frame that does not is skipped every shot; says so unless quiet."""
        scanned = {x.key: np.asarray(x.values, dtype=float) for x in xvars}
        for key, img in self.images().items():
            s = img.t_stage
            if s is None:
                img.t_stage_s, img.in_stage = None, True
                continue
            if isinstance(s, str):
                if s in scanned:
                    v, what = float(np.min(scanned[s])), f"'{s}' (scanned; shortest)"
                elif params is not None and hasattr(params, s):
                    v, what = float(getattr(params, s)), f"'{s}'"
                else:
                    raise ValueError(f"image {key!r}: t_stage {s!r} is not a parameter")
            else:
                v, what = float(s), "t_stage"
            img.t_stage_s = v
            img.in_stage = abs(img.t_offset) <= v
            if not img.in_stage and not quiet and self.enabled:
                print(f"[diag cams] {key}: not taken this run -- |t_offset| = {abs(img.t_offset) * 1e3:g} ms "
                      f"is longer than {what} = {v * 1e3:g} ms")

    def record(self):
        """Every image as this run takes it (for the run file)."""
        # "camera" was the frame's camera attribute name before 2026-10-20; it is
        # the camera entry's key now, as "camera_id" always was
        return {"enabled": self.enabled,
                "default_imaging_type": next(
                    (n for n, v in vars(img_types).items()
                     if v == getattr(self, 'default_imaging_type', None)), None),
                "images": {k: {"camera": img.camera.key, "camera_id": img.camera.key,
                               "frame": img.j, "t_offset": img.t_offset,
                               "t_stage": img.t_stage, "t_stage_s": img.t_stage_s,
                               "taken": bool(img.live and img.in_stage
                                             and _connected(img._stream)),
                               **img.settings(),
                               "from_camera_entry": [s for s in SETTING_KEYS
                                                     if s not in img.given],
                               **_exposure_delay_record(img)}
                           for k, img in self.images().items()}}

    def finish(self):
        """Drain and close every stream (end_wax does this itself)."""
        for cs in self.streams().values():
            try:
                cs.finish()
            except Exception:
                pass
