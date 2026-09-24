"""Cameras from the command line: what liveOD holds, release one that no run is
using, hand it back, or pull a frame off a Basler.

liveOD keeps the lab's cameras open between runs, and a Basler is USB-exclusive:
while liveOD has it, nothing else (the beacon Basler server, a script) can open
it.  This tool closes such a camera in liveOD -- the server refuses if the run in
progress is using it -- and, for ``grab``, then pulls one frame through the beacon
Basler server (``beacon.basler.frame_grabber``).

    python -m waxx.util.live_od.camera_cli list
    python -m waxx.util.live_od.camera_cli release xy_basler
    python -m waxx.util.live_od.camera_cli open xy_basler
    python -m waxx.util.live_od.camera_cli grab xy_basler --release --reopen --save shot.png

``grab`` without ``--release`` refuses if liveOD has the camera open, so nobody
loses a camera by accident.  ``--reopen`` gives it back to liveOD afterwards (only
if this command released it).  Without a reachable liveOD, ``grab`` goes straight
to the beacon server, matching the camera by its DeviceUserID / serial.

Exit codes: 0 ok; 1 error; 2 refused (run in progress uses the camera, or liveOD
holds it and no --release); 3 no server reachable; 4 no frame within the timeout.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_UNREACHABLE, EXIT_TIMEOUT = 0, 1, 2, 3, 4


def _connect(discovery_timeout: float):
    from waxx.util.live_od.live_od_client import LiveODClient
    return LiveODClient(discovery_timeout=discovery_timeout)


def _print_cameras(client) -> None:
    status = client.poll()
    cams = status.get("cameras")
    if cams is None:
        print("this liveOD server does not report camera state (restart it with current code)",
              file=sys.stderr)
        return
    run = status.get("run_in_progress")
    print(f"liveOD: run_in_progress={run} run_id={status.get('run_id')} "
          f"run_camera={status.get('run_camera_key') or '-'}")
    for key, info in cams.items():
        tag = "  <- in use by the run" if run and key == status.get("run_camera_key") else ""
        print(f"  {key:14s} {info.get('state', '?'):9s} {info.get('camera_type', ''):7s} "
              f"{info.get('serial_no', '')}{tag}")


def cmd_list(args) -> int:
    try:
        client = _connect(args.discovery_timeout)
    except Exception as exc:
        print(f"liveOD not reachable: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    _print_cameras(client)
    return EXIT_OK


def cmd_release(args) -> int:
    try:
        client = _connect(args.discovery_timeout)
    except Exception as exc:
        print(f"liveOD not reachable: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    try:
        before = client.cameras().get(args.camera, {}).get("state")
        info = client.release_camera(args.camera, timeout=args.timeout)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except RuntimeError as exc:          # the server refused, or the close failed
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED if "rejected" in str(exc) else EXIT_ERROR
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_TIMEOUT
    verb = "already closed" if before == "closed" else "released (closed in liveOD)"
    print(f"{args.camera}: {verb}; state={info.get('state')} serial={info.get('serial_no', '')}")
    return EXIT_OK


def cmd_open(args) -> int:
    try:
        client = _connect(args.discovery_timeout)
    except Exception as exc:
        print(f"liveOD not reachable: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    try:
        info = client.open_camera(args.camera, timeout=args.timeout)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED if "rejected" in str(exc) else EXIT_ERROR
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_TIMEOUT
    print(f"{args.camera}: state={info.get('state')}")
    return EXIT_OK


def cmd_grab(args) -> int:
    from beacon.basler.frame_grabber import (
        CameraBusy, CameraNotFound, FrameGrabError, FrameTimeout, grab_frame, save_frame,
    )

    query = args.camera
    client = None
    released = False
    try:
        client = _connect(args.discovery_timeout)
    except Exception as exc:
        print(f"note: liveOD not reachable ({exc}); grabbing straight from the beacon server",
              file=sys.stderr)

    if client is not None:
        try:
            cams = client.cameras()
        except LookupError as exc:
            print(f"note: {exc}; grabbing straight from the beacon server", file=sys.stderr)
            cams = {}
        entry = cams.get(args.camera)
        if entry is not None:
            ctype = (entry.get("camera_type") or "").lower()
            if ctype and ctype != "basler":
                print(f"{args.camera} is a {ctype}, not a Basler; only Baslers have a beacon "
                      f"server to grab from", file=sys.stderr)
                return EXIT_ERROR
            query = entry.get("serial_no") or args.camera
            state = entry.get("state")
            if state != "closed":
                if not args.release:
                    print(f"liveOD holds {args.camera} (state {state}). Pass --release to close "
                          f"it there first (refused if the run in progress is using it).",
                          file=sys.stderr)
                    return EXIT_REFUSED
                try:
                    client.release_camera(args.camera, timeout=args.timeout)
                    released = True
                    print(f"{args.camera}: released from liveOD")
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return EXIT_REFUSED if "rejected" in str(exc) else EXIT_ERROR
                except TimeoutError as exc:
                    print(str(exc), file=sys.stderr)
                    return EXIT_TIMEOUT
        elif cams:
            print(f"note: liveOD has no camera {args.camera!r} (it has {sorted(cams)}); "
                  f"asking the beacon server by that name", file=sys.stderr)

    rc = EXIT_OK
    try:
        fr = grab_frame(query, timeout=args.frame_timeout, trigger_mode=args.trigger_mode,
                        fresh=not args.latest, collect_for=args.collect_for)
        print(fr.summary())
        if args.save:
            print(f"  saved {save_frame(fr, args.save)}")
    except CameraNotFound as exc:
        print(f"camera not found: {exc}", file=sys.stderr)
        rc = EXIT_UNREACHABLE
    except CameraBusy as exc:
        print(f"camera busy: {exc}", file=sys.stderr)
        rc = EXIT_REFUSED
    except FrameTimeout as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        rc = EXIT_TIMEOUT
    except FrameGrabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        rc = EXIT_ERROR
    finally:
        if released and args.reopen and client is not None:
            # A moment for the beacon server to let go of the USB device.
            time.sleep(0.5)
            try:
                info = client.open_camera(args.camera, timeout=args.timeout)
                print(f"{args.camera}: back in liveOD (state {info.get('state')})")
            except Exception as exc:
                print(f"could not reopen {args.camera} in liveOD: {exc}", file=sys.stderr)
                rc = rc or EXIT_ERROR
        elif released and not args.reopen:
            print(f"{args.camera} is still closed in liveOD (open it from the GUI, or "
                  f"`camera_cli open {args.camera}`); the next run that needs it reopens it.")
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m waxx.util.live_od.camera_cli",
        description="See, release, or reopen the cameras liveOD holds; grab a frame from a Basler.")
    ap.add_argument("--discovery-timeout", type=float, default=5.0,
                    help="seconds to wait for the liveOD beacon (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="seconds to wait for liveOD to finish an open/close (default %(default)s)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="cameras on liveOD's bar, their state, and the run's camera")

    p = sub.add_parser("release", help="close a camera in liveOD so another process can open it")
    p.add_argument("camera", help="liveOD camera key, e.g. xy_basler")

    p = sub.add_parser("open", help="(re)open a camera in liveOD")
    p.add_argument("camera", help="liveOD camera key, e.g. xy_basler")

    p = sub.add_parser("grab", help="pull one fresh frame from a Basler via its beacon server")
    p.add_argument("camera", help="liveOD camera key (or serial / DeviceUserID if liveOD is down)")
    p.add_argument("--release", action="store_true",
                   help="if liveOD holds the camera, close it there first")
    p.add_argument("--reopen", action="store_true",
                   help="give the camera back to liveOD after the grab (if this command released it)")
    p.add_argument("--frame-timeout", type=float, default=None,
                   help="seconds to wait for a fresh frame (default 10 in trigger mode, 3 in free run)")
    p.add_argument("--trigger-mode", choices=["On", "Off"], default=None,
                   help="set the camera's trigger mode for this grab (restored afterwards)")
    p.add_argument("--latest", action="store_true",
                   help="accept the frame the beacon server already has instead of waiting for a fresh one")
    p.add_argument("--collect-for", type=float, default=2.0,
                   help="seconds to listen for beacon Basler servers (default %(default)s)")
    p.add_argument("--save", metavar="PATH", default=None,
                   help="write the frame to PATH (.npy, .png, .tif, ...)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return {"list": cmd_list, "release": cmd_release, "open": cmd_open, "grab": cmd_grab}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
