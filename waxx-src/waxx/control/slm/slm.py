import socket
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
