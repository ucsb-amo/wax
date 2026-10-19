"""Per-shot diagnostic frames: a frame of named cameras and images, and the
lookup of a camera's trigger TTL on an experiment's ttl frame.

A lab's ``camera_id`` subclasses ``diagnostic_frame`` and assigns its
cameras and images between ``setup()`` and ``cleanup()``, like the device
frames::

    class diagnostic_frame(diagnostic_frame_waxx):
        def __init__(self):
            self.setup()
            self.mot = self.camera(cameras.mot_basler, exposure_time=300e-6, gain=3., roi=[...])
            self.img_mot = self.image(self.mot, t=-150e-3, t_stage='t_mot_load', show='image')
            self.cleanup()

An image is one frame its camera takes every shot. Its key -- the DataVault
key it is saved under -- is the attribute name. Its frame index is its place
among its camera's images in the order they are assigned: assign them in the
order the sequence takes them. The sequence takes it where it belongs with
``self.diag.img_mot.trigger()``: ``t`` (s) from the timeline cursor
(negative: before it); the cursor stays where it was. Nothing is taken with
diagnostics off, during warm-up shots (``active`` cleared), or for a camera
that could not be used this run (a placeholder stream).

``t_stage`` (optional) is the length of the stage the frame sits in, in s or
as the name of an ExptParams attribute: a frame with ``|t| > t_stage`` would
fall outside its stage -- before it starts, possibly in the past -- and is
skipped every shot, which is said at prepare. It is resolved from the run's
params when the run is bound and again at finish_prepare (``resolve_stages``);
a scanned param counts by its shortest value.

For a run, the lab builds a fresh frame and binds it (``bind``) with its own
stream factory and TTL lookup. Each image then holds its camera's trigger TTL
and reports its edge to its camera's stream from the host, so every image is
the same kind of object to the ARTIQ compiler and no per-camera code is made.
"""

import numpy as np

from artiq.experiment import kernel, rpc, delay, now_mu

from waxx.control.cameras.camera_stream_client import (
    InactiveCameraStream, TriggeredCameraStreamClient)

_SHOW = ('image', 'counts', None)


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


def _number(v):
    return float(f"{float(v):.4g}")


class DiagnosticCamera:
    """A camera of the diagnostics: its camera entry, the exposure_time (s),
    gain (dB) and viewer ROI ``[x1, y1, x2, y2]`` it runs at (None = the
    camera's saved defaults), its images (in frame-index order) and the
    ``label`` the diagnostic viewer names it by."""

    def __init__(self, camera, exposure_time=None, gain=None, roi=None, label=None):
        self.camera = camera
        self.exposure_time = exposure_time
        self.gain = gain
        self.roi = [int(v) for v in roi] if roi else None
        self._label = label
        self.images = []
        self.key = ""           # the attribute name, set by the frame
        self.stream = None      # set by bind

    @property
    def label(self):
        return self._label or str(getattr(self.camera, 'key', '') or self.key).replace('_', ' ')

    @property
    def frames(self):
        """The frame keys, in frame-index order."""
        return [img.key for img in self.images]

    @property
    def frame_settings(self):
        """{frame key: its own exposure / gain} for the frames that have them."""
        return {img.key: img.settings() for img in self.images if img.settings()}

    def frame_index(self, key):
        """The index of frame ``key`` on this camera, -1 if it takes none."""
        return self.frames.index(key) if key in self.frames else -1

    def assignment(self, **values):
        """This camera's line in a frame's ``__init__`` (``values`` replace
        exposure_time / gain / roi), then one line per image."""
        cam_key = getattr(self.camera, 'key', '')
        args = [f"cameras.{cam_key}" if cam_key else repr(self.camera)]
        vals = {"exposure_time": self.exposure_time, "gain": self.gain, "roi": self.roi}
        vals.update(values)
        for k in ("exposure_time", "gain"):
            if vals[k] is not None:
                args.append(f"{k}={_number(vals[k])!r}")
        if vals["roi"] is not None:
            args.append(f"roi={[int(v) for v in vals['roi']]!r}")
        if self._label:
            args.append(f"label={self._label!r}")
        lines = [f"self.{self.key} = self.camera(" + ", ".join(args) + ")"]
        return "\n".join(lines + [img.assignment() for img in self.images])


class DiagnosticImage:
    """One frame a camera takes every shot (module docstring): ``t`` (s)
    from where ``trigger()`` is called, ``t_stage`` (s or an ExptParams name;
    None: no limit), its own ``exposure_time`` (s) / ``gain`` (dB) (None:
    the camera's), and for the diagnostic viewer a ``label`` and ``show``
    ('image', 'counts' or None: not shown)."""

    def __init__(self, camera, t=0., t_stage=None, exposure_time=None, gain=None,
                 label=None, show='counts'):
        if not isinstance(camera, DiagnosticCamera):
            raise TypeError(f"DiagnosticImage: camera must be a DiagnosticCamera, not {camera!r}")
        if show not in _SHOW:
            raise ValueError(f"DiagnosticImage: show={show!r}, not one of {_SHOW}")
        if not (t_stage is None or isinstance(t_stage, str)):
            t_stage = float(t_stage)
        self.camera = camera
        self.t = float(t)
        self.t_stage = t_stage
        self.exposure_time = None if exposure_time is None else float(exposure_time)
        self.gain = None if gain is None else float(gain)
        self._label = label
        self.show = show
        self.key = ""               # the attribute name, set by the frame
        camera.images.append(self)
        # kernel side, set by bind (the same types for every image)
        self.frame = None           # the run's frame: frame.active
        self.core = None
        self.ttl = None             # the camera's trigger TTL
        self.j = -1                 # frame index on the camera
        self.live = False           # the camera's stream takes frames this run
        self.in_stage = True        # |t| <= t_stage (resolve_stages)
        self.exposure_delay = 0.
        self.t_trigger = 2.e-6
        self._stream = None
        self.t_stage_s = None       # t_stage in s, as resolved

    @property
    def label(self):
        return self._label or self.key.removeprefix('img_').replace('_', ' ')

    def settings(self):
        """This frame's own exposure / gain ({} when it has none)."""
        return {k: v for k, v in (('exposure_time', self.exposure_time),
                                  ('gain', self.gain)) if v is not None}

    def assignment(self):
        """This image's line in a frame's ``__init__``."""
        args = [f"self.{self.camera.key}", f"t={self.t!r}"]
        if self.t_stage is not None:
            args.append(f"t_stage={self.t_stage!r}")
        args += [f"{k}={v!r}" for k, v in self.settings().items()]
        if self._label:
            args.append(f"label={self._label!r}")
        if self.show != 'counts':
            args.append(f"show={self.show!r}")
        return f"self.{self.key} = self.image(" + ", ".join(args) + ")"

    @kernel
    def trigger(self):
        """Take this frame ``t`` from the cursor; the cursor stays."""
        if self.frame.active and self.live and self.in_stage:
            delay(self.t)
            self.announce_edge(now_mu() - self.core.get_rtio_counter_mu())
            delay(-self.exposure_delay)
            self.ttl.pulse(self.t_trigger)
            delay(self.exposure_delay - self.t_trigger - self.t)

    @rpc(flags={"async"})
    def announce_edge(self, slack_mu):
        """Async RPC target: the edge, to the camera's stream (it pairs the
        frame with the shot and tells its trigger-line siblings)."""
        try:
            self._stream.announce_trigger_mu(slack_mu, self.j)
        except Exception:
            pass


class diagnostic_frame:
    """The base of a lab's frame of diagnostic cameras and images (module
    docstring). Cameras and images are named by the attributes they are
    assigned to."""

    enabled = False
    active = False          # kernel switch: cleared around warm-up shots

    def __init__(self):
        self.setup()
        self.cleanup()

    def setup(self):
        # a camera with no saved viewer ROI stores its frames block-averaged this much
        self.no_roi_downsample = 4

    def cleanup(self):
        self._check()

    def __setattr__(self, name, value):
        if isinstance(value, (DiagnosticCamera, DiagnosticImage)):
            if hasattr(diagnostic_frame, name):
                raise ValueError(f"{name!r} is taken by diagnostic_frame.{name}")
            if value.key and value.key != name:
                raise ValueError(f"{type(self).__name__}.{name} is already .{value.key}")
            if isinstance(value, DiagnosticImage) and not name.isidentifier():
                raise ValueError(f"image key {name!r} is not an identifier")
            value.key = name
        object.__setattr__(self, name, value)

    def camera(self, camera, exposure_time=None, gain=None, roi=None, label=None):
        """A camera (assign it to an attribute: its name)."""
        return DiagnosticCamera(camera, exposure_time=exposure_time, gain=gain, roi=roi,
                                label=label)

    def image(self, camera, t=0., t_stage=None, exposure_time=None, gain=None, label=None,
              show='counts'):
        """An image ``camera`` takes every shot (assign it to an attribute:
        its key). Assign a camera's images in the order they are taken."""
        return DiagnosticImage(camera, t=t, t_stage=t_stage, exposure_time=exposure_time,
                               gain=gain, label=label, show=show)

    def entries(self):
        """The cameras, in the order assigned."""
        return [v for v in vars(self).values() if isinstance(v, DiagnosticCamera)]

    def images(self):
        """{key: image}, in the order assigned."""
        return {k: v for k, v in vars(self).items() if isinstance(v, DiagnosticImage)}

    def _check(self):
        cams = self.entries()
        for img in self.images().values():
            if img.camera not in cams:
                raise ValueError(f"image {img.key!r}: its camera is not an attribute of the frame")
            if img.camera.images.count(img) != 1:
                raise ValueError(f"image {img.key!r}: listed {img.camera.images.count(img)} "
                                 f"times on camera {img.camera.key!r}")
        for cam in cams:
            unnamed = [img for img in cam.images if not img.key]
            if unnamed:
                raise ValueError(f"camera {cam.key!r}: {len(unnamed)} image(s) not assigned to an "
                                 f"attribute of the frame")

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
            streams = make_streams(expt, [
                dict(camera=c.camera, key=c.frames, mode='triggered', on_error='dummy',
                     exposure_time=c.exposure_time, gain=c.gain, roi=c.roi,
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
                if live:
                    img.exposure_delay = float(stream.exposure_delay)
                    img.t_trigger = float(stream.t_trigger)
        self.resolve_stages(getattr(expt, 'params', None), quiet=True)
        if self.enabled:
            up = [c.key for c in cams if isinstance(c.stream, TriggeredCameraStreamClient)]
            down = [c.key for c in cams if c.key not in up]
            print("[diag cams] connected: " + (", ".join(up) or "none")
                  + (f" | NOT connected: {', '.join(down)}" if down else ""))
        return self

    def streams(self):
        """{camera key: its stream} (bound cameras with images)."""
        return {c.key: c.stream for c in self.entries() if c.stream is not None}

    def resolve_stages(self, params, xvars=(), quiet=False):
        """Each image's ``t_stage`` in s from ``params`` (a scanned key -- in
        ``xvars`` -- by its shortest value) and whether ``|t|`` fits in it.
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
            img.in_stage = abs(img.t) <= v
            if not img.in_stage and not quiet and self.enabled:
                print(f"[diag cams] {key}: not taken this run -- |t| = {abs(img.t) * 1e3:g} ms "
                      f"is longer than {what} = {v * 1e3:g} ms")

    def record(self):
        """Every image as this run takes it (for the run file)."""
        return {"enabled": self.enabled,
                "images": {k: {"camera": img.camera.key,
                               "camera_id": str(getattr(img.camera.camera, 'key', '')),
                               "frame": img.j, "t": img.t,
                               "t_stage": img.t_stage, "t_stage_s": img.t_stage_s,
                               "taken": bool(img.live and img.in_stage),
                               **img.settings()}
                           for k, img in self.images().items()}}

    def finish(self):
        """Drain and close every stream (end_wax does this itself)."""
        for cs in self.streams().values():
            try:
                cs.finish()
            except Exception:
                pass
