import socket
import time
from artiq.coredevice.core import Core
from artiq.language.core import now_mu, delay, kernel
from waxx.config.expt_params import ExptParams
from waxx.control.slm import slm_link
from waxx.util import console
import numpy as np
import json
di = -1
dv = 1.
dm = 1
SLM_RPC_DELAY = 0.25
# A write waits for the server to say the mask is on the display. A reinit the
# monitor server asked for just before this run can be ahead of it in the
# server's queue (seconds), hence the long limit.
SLM_APPLIED_TIMEOUT_S = 30.
# A server that has not said "queued" by then is one from before replies.
SLM_FIRST_REPLY_S = 3.
# A run about to start waits this long for a due reinit (reinit_if_due). The
# supervisor on the SLM PC calls one SLM task hung after 2 min.
SLM_RUN_START_REINIT_TIMEOUT_S = 90.
# How often it asks the server whether the reinit is done.
SLM_REINIT_POLL_S = 0.5
# A status query's connect timeout (the server answers status at once).
SLM_STATUS_CONNECT_S = 2.
# The status fields a run keeps from before and after its reinit.
_STATUS_KEYS = ("instance", "start_count", "pattern_epoch", "reinit_due", "due_for_s",
                "next_due_in_s", "reinit_in_progress", "slm_ready", "last_reinit_error",
                "queue_len", "pattern")


class SLMWriteError(RuntimeError):
    """The SLM server could not be reached, refused the mask, dropped it, or
    did not report it applied: the mask on the SLM is not the one asked for."""


class SLM:
    def __init__(self, expt_params=ExptParams(), core=Core,
                 server_ip=slm_link.SLM_HOST, server_port=slm_link.SLM_PORT):
        self.server_ip = server_ip
        self.server_port = server_port
        self.params = expt_params
        self.core = core
        # None until the first write: whether the server answers (servers from
        # before 2026-09-26 apply a mask without a word).
        self._server_replies = None

    def write_phase_mask(self, dimension=dv, phase=dv, x_center=di, y_center=di, mask_type='spot', initialize=False,
                         verbose=True):
        """Writes a phase spot of given dimension and phase to the specified
        position on the slm display.

        Args:
            dimension (float): Dimnesion (in m) of the phase mask. If set to
            zero, gives uniform phase pattern. Defaults to
            ExptParams.dimension_slm_mask.
            phase (float): Phase (in radians) for the phase mask. Defaults to
            ExptParams.phase_slm_mask.
            x_center (int): Horizontal position (in pixels) of the
            phase spot (from top right). Indexed from 1 to 1920. Defaults to
            ExptParams.px_slm_phase_mask_position_x.
            y_center (int): Vertical position (in pixels) of the
            phase spot (from top right). Indexed from 1 to 1200. Defaults to
            ExptParams.px_slm_phase_mask_position_y. 
            mask_type (str): The type of mask. It can be spot, grating or cross. 
            Defaults to ExptParams.slm_mask.
            initialize (booling): True for doing initialization on client side, and False
            for letting SLM self-reinitialze automatically.

        Returns once the server reports the mask on the display. Raises
        SLMWriteError when it cannot say so (unreachable, error, dropped, no
        "applied" in time): the run must not go on with an unknown mask. A
        server from before replies gets one warning and the old
        fire-and-forget send.
        """
        if dimension == dv:
            dimension = self.params.dimension_slm_mask
        if phase == dv:
            phase = self.params.phase_slm_mask
        if x_center == di:
            x_center = self.params.px_slm_phase_mask_position_x
        if y_center == di:
            y_center = self.params.px_slm_phase_mask_position_y

        x_center = int(x_center)
        y_center = int(y_center)

        if mask_type == 'spot':
            mask = 'spot'
        elif mask_type == 'grating':
            mask = 'grating'
        elif mask_type == 'cross':
            mask = 'cross'
        else:
            raise ValueError("mask_type must be one of 'spot', 'grating', or 'cross'.")

        dimension = int(dimension * 1.e6)
        command = {
                "mask": mask,
                "center": [x_center, y_center],
                "phase": phase/np.pi,
                "dimension": dimension,
                "initialize": initialize,
                "spacing": 10,
                "angle": 45,
            }
        # command = f"{int(dimension)} {phase/np.pi} {x_center} {y_center} {mask}"
        applied = self._write(command)
        if verbose:
            console.info(f"[slm] {mask_type}: {dimension} um, "
                         f"{phase/np.pi:.2f} pi @ ({x_center}, {y_center})"
                         + ("" if applied is None
                            else f", applied in {applied.get('t_apply_s', float('nan')):.3f} s"))
            console.info(f"[slm] sent: {command}", level=console.VERBOSE)

    def _write(self, command):
        """Send one pattern; the server's "applied" reply, or None when the
        server does not reply (see write_phase_mask)."""
        if self._server_replies is False:
            try:
                self._send_command(command)
            except OSError as e:
                raise self._failed(f"could not reach the SLM server at "
                                   f"{self.server_ip}:{self.server_port}: {e}")
            return None
        try:
            reply = slm_link.exchange(self.server_ip, self.server_port, command,
                                      until=("applied", "error", "dropped"),
                                      first_reply_s=SLM_FIRST_REPLY_S,
                                      total_s=SLM_APPLIED_TIMEOUT_S)
        except slm_link.NoReply:
            # The pattern went out; an old server applies it without a word.
            self._server_replies = False
            print(f"[slm] WARNING: the SLM server at {self.server_ip}:{self.server_port} "
                  "does not confirm writes (a server from before 2026-09-26): masks are "
                  "sent unconfirmed for the rest of this run, as before.")
            return None
        except (slm_link.ReplyTimeout, OSError) as e:
            raise self._failed(f"{e} ({self.server_ip}:{self.server_port})")
        self._server_replies = True
        if reply.get("status") != "applied":
            raise self._failed(f"the SLM server {reply.get('status')} the mask"
                               + (f": {reply['error']}" if reply.get("error") else ""))
        return reply

    @staticmethod
    def _failed(text) -> SLMWriteError:
        # Printed too: an exception raised through a kernel loses its message.
        print(f"[slm] ERROR: {text} -- the mask on the SLM is not the one asked for.")
        return SLMWriteError(text)

    # --- the reinit, at a run boundary --------------------------------------------

    def reinit_if_due(self, by="a run starting", required=False, timeout_s=None,
                      poll_s=None) -> dict:
        """Have a due SLM reinit done now, before a run starts, and wait for it.

        The SLM server re-initialises only when asked: once an hour it marks a
        reinit due, and the monitor server asks for it while the machine is
        idle -- which back-to-back runs, or a run loop, may never leave it. So
        a run about to start asks the server:

        * a reinit due: ask for it, wait until it is done;
        * one in progress (the monitor server's, say): wait until it is done;
        * none: return at once (one status query).

        Done means a new ``pattern_epoch`` (the server puts the last pattern
        back after a reinit) or a new server process (``instance``: its
        start-up initialisation is one). Waits at most `timeout_s`.

        `required` (a run that uses the SLM): raise :class:`SLMWriteError`
        when the reinit failed or did not finish, so the run does not start on
        an SLM in an unknown state. A server that is unreachable, or too old
        to take control commands, never stops a run here: its mask writes
        report that themselves.

        Returns what happened, for the run's file: ``{"result", "by", "t_s",
        "before", "after", "error"}``, ``result`` one of ``"not_due"``,
        ``"reinit_done"``, ``"waited"`` (one already in progress finished),
        ``"restarted"``, ``"failed"``, ``"timeout"``, ``"no_control"`` (a
        server from before 2026-09-28, which re-initialises by itself, blank),
        ``"unreachable"``.
        """
        timeout_s = SLM_RUN_START_REINIT_TIMEOUT_S if timeout_s is None else timeout_s
        poll_s = SLM_REINIT_POLL_S if poll_s is None else poll_s
        t0 = time.monotonic()
        record = {"result": "", "by": by, "t_s": 0.0, "before": None, "after": None,
                  "error": ""}

        def finish(result, error="", after=None):
            record.update(result=result, error=error, t_s=round(time.monotonic() - t0, 2))
            if after is not None:
                record["after"] = {k: after.get(k) for k in _STATUS_KEYS}
            return record

        where = f"{self.server_ip}:{self.server_port}"
        try:
            st = self._status()
        except slm_link.NoReply:
            print(f"[slm] WARNING: the SLM server at {where} takes no control commands (a "
                  f"server from before 2026-09-28): it re-initialises by itself once an "
                  f"hour and then shows a BLANK mask, whether a run is going or not.")
            return finish("no_control")
        except (slm_link.ReplyTimeout, OSError) as e:
            print(f"[slm] WARNING: could not ask the SLM server at {where} whether a reinit "
                  f"is due ({e}); starting without.")
            return finish("unreachable", str(e))
        if st.get("status") != "ok":
            print(f"[slm] WARNING: the SLM server at {where} refused a status query "
                  f"({st.get('error')}); starting without checking for a reinit.")
            return finish("unreachable", f"status refused: {st.get('error')}")
        record["before"] = {k: st.get(k) for k in _STATUS_KEYS}
        if not (st.get("reinit_due") or st.get("reinit_in_progress")):
            return finish("not_due")

        epoch0, instance0 = st.get("pattern_epoch"), st.get("instance")
        if st.get("reinit_in_progress"):
            done_result = "waited"
            console.info("[slm] the SLM is re-initialising: this run waits for it.")
        else:
            done_result = "reinit_done"
            due = st.get("due_for_s")
            console.info("[slm] an SLM reinit is due"
                         + (f" (for {due / 60:.0f} min)" if isinstance(due, (int, float)) else "")
                         + ": re-initialising it before this run starts.")
            try:
                reply = slm_link.exchange(self.server_ip, self.server_port,
                                          {"cmd": "reinit", "by": by}, control=True,
                                          until=("queued", "error"),
                                          connect_s=SLM_STATUS_CONNECT_S,
                                          first_reply_s=SLM_FIRST_REPLY_S,
                                          total_s=SLM_FIRST_REPLY_S)
            except (slm_link.NoReply, slm_link.ReplyTimeout, OSError) as e:
                return self._run_start_reinit_failed(
                    finish("failed", f"the reinit request failed: {e}"), required)
            if reply.get("status") != "queued":
                return self._run_start_reinit_failed(
                    finish("failed", f"the SLM server refused the reinit: {reply.get('error')}"),
                    required)

        # The request went out and closed ("queued"): the server serves one
        # connection at a time, so it is followed by status queries, which
        # also leave the monitor server's polls through.
        last, idle_polls = st, 0
        while time.monotonic() - t0 < timeout_s:
            time.sleep(poll_s)
            try:
                s2 = self._status()
            except (slm_link.NoReply, slm_link.ReplyTimeout, OSError):
                continue            # busy with another client, or restarting
            if s2.get("status") != "ok":
                continue
            last = s2
            if s2.get("reinit_in_progress"):
                idle_polls = 0
                continue
            if s2.get("instance") != instance0 or s2.get("pattern_epoch") != epoch0:
                if s2.get("slm_ready") is False:
                    return self._run_start_reinit_failed(
                        finish("failed", f"the SLM is not ready: {s2.get('slm_not_ready')}",
                               after=s2), required)
                result = "restarted" if s2.get("instance") != instance0 else done_result
                finish(result, after=s2)
                pat = s2.get("pattern") or {}
                console.info(f"[slm] re-initialised in {record['t_s']:.1f} s"
                             + (" (a new SLM server process)" if result == "restarted" else "")
                             + f"; pattern put back ({pat.get('mask')} "
                             f"{pat.get('dimension')} um @ ({pat.get('center_x')}, "
                             f"{pat.get('center_y')})). The run starts now.")
                return record
            # Nothing running and no new epoch: finished without success once
            # nothing is queued either -- asked twice, for the moment between
            # the worker taking the reinit off the queue and starting it.
            idle_polls = 0 if s2.get("queue_len") else idle_polls + 1
            if idle_polls >= 2:
                return self._run_start_reinit_failed(
                    finish("failed", s2.get("last_reinit_error")
                           or "the reinit ended without re-initialising the SLM", after=s2),
                    required)
        return self._run_start_reinit_failed(
            finish("timeout", f"not re-initialised within {timeout_s:g} s", after=last),
            required)

    def _status(self) -> dict:
        return slm_link.exchange(self.server_ip, self.server_port, {"cmd": "status"},
                                 control=True, until=("ok", "error"),
                                 connect_s=SLM_STATUS_CONNECT_S,
                                 first_reply_s=SLM_FIRST_REPLY_S, total_s=SLM_FIRST_REPLY_S)

    @staticmethod
    def _run_start_reinit_failed(record, required) -> dict:
        text = f"the SLM reinit before this run did not complete ({record['result']}): " \
               f"{record['error']}"
        if required:
            print(f"[slm] ERROR: {text}. This run uses the SLM, so it does not start.")
            raise SLMWriteError(text)
        print(f"[slm] WARNING: {text}. This run does not use the SLM, so it starts anyway.")
        return record

    @kernel
    def write_phase_mask_kernel(self, dimension=dv, phase=dv, x_center=di, y_center=di, mask_type='spot',initialize=False,
                                verbose=True):
        """Writes a phase spot of given dimension and phase to the specified
        position on the slm display.

        Args:
            dimension (float): Dimnesion (in m) of the phase mask. If set to
            zero, gives uniform phase pattern. Defaults to
            ExptParams.dimension_slm_mask.
            phase (float): Phase (in radians) for the phase mask. Defaults to
            ExptParams.phase_slm_mask.
            x_center (int): Horizontal position (in pixels) of the
            phase spot (from top right). Indexed from 1 to 1920. Defaults to
            ExptParams.px_slm_phase_mask_position_x.
            y_center (int): Vertical position (in pixels) of the
            phase spot (from top right). Indexed from 1 to 1200. Defaults to
            ExptParams.px_slm_phase_mask_position_y. 
            mask_type (str): The type of mask. It can be spot, grating or cross. 
            Defaults to ExptParams.slm_mask.
            initialize (booling): True for doing initialization on client side, and 
            False for letting SLM self-reinitialze automatically. 
        """    
        self.core.wait_until_mu(now_mu())
        self.write_phase_mask(dimension, phase, x_center, y_center, mask_type, initialize, verbose)
        # The RPC returns once the mask is on the display, which can take
        # seconds (a reinit ahead of it); that time is not on the timeline.
        self.core.break_realtime()
        delay(SLM_RPC_DELAY)

    def _send_command(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.server_ip, self.server_port))
            if isinstance(command, dict):
                command = json.dumps(command) # Convert dict to JSON string
            client_socket.sendall(command.encode('utf-8'))

    # def _launch_pattern_gui(self):
    #     root = tk.Tk()
    #     root.withdraw()
    #     Patternapp(root)
    #     root.mainloop()

if __name__ == '__main__':
    slm = SLM()
