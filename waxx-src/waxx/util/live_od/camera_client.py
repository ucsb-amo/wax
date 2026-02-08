"""
Experiment-side client for the camera server.

Subclasses ``CommClient`` and adds a binary protocol layer (length-prefixed
pickle) for sending complex objects (camera params, xvar dicts, etc.) over a
persistent TCP connection that lasts for one experiment run.

Typical usage inside an ARTIQ experiment::

    from waxx.util.live_od.camera_client import CameraClient

    client = CameraClient(CAMERA_SERVER_IP, CAMERA_SERVER_PORT)
    client.connect()

    # -- handshake (blocks until camera is connected & grab loop running) --
    reply = client.send_new_run(
        camera_params=self.camera_params,
        data_filepath=self.data_filepath,
        save_data=self.run_info.save_data,
        N_img=self.params.N_img,
        N_shots=self.params.N_shots,
        N_pwa_per_shot=self.params.N_pwa_per_shot,
        imaging_type=self.run_info.imaging_type,
        run_id=self.run_info.run_id,
    )
    assert reply["cmd"] == "camera_ready"  # camera connected, grab loop started

    # -- run experiment (trigger camera, etc.) --
    ...

    # -- send xvars for the current shot --
    client.send_xvars({"t_tof": 20e-3, "freq_tweezer": 80e6})

    # -- done --
    client.send_run_complete()
    client.disconnect()
"""

import socket
from waxx.util.comms_server.comm_client import CommClient
from waxx.util.live_od.protocol import send_msg, recv_msg


class CameraClient(CommClient):
    """
    Client for communicating with the camera server from an ARTIQ experiment.

    Inherits the simple ``send_message(str)`` interface from ``CommClient``
    and adds binary-protocol methods for the camera handshake.

    Parameters
    ----------
    server_ip : str
        Camera server IP address.
    server_port : int
        Camera server command port.
    """

    def __init__(self, server_ip, server_port):
        super().__init__(server_ip, server_port)
        self._persistent_sock: socket.socket | None = None

    # ------------------------------------------------------------------
    #  Persistent connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Open a persistent TCP connection for the duration of one run."""
        self._persistent_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._persistent_sock.connect(self.server_address)

    def disconnect(self):
        """Close the persistent connection."""
        if self._persistent_sock is not None:
            try:
                self._persistent_sock.close()
            except Exception:
                pass
            self._persistent_sock = None

    # ------------------------------------------------------------------
    #  Low-level binary helpers
    # ------------------------------------------------------------------

    def _send(self, msg):
        send_msg(self._persistent_sock, msg)

    def _recv(self):
        return recv_msg(self._persistent_sock)

    def _send_recv(self, msg):
        """Send *msg* and block until the server replies."""
        self._send(msg)
        return self._recv()

    # ------------------------------------------------------------------
    #  High-level run commands
    # ------------------------------------------------------------------

    def send_new_run(self, camera_params, data_filepath, save_data,
                     N_img, N_shots, N_pwa_per_shot, imaging_type, run_id):
        """
        Send camera params and run metadata to the server.

        Blocks until the server has connected the camera and started the
        grab loop. Returns ``{"cmd": "camera_ready"}`` on success —
        at that point the experiment can proceed to trigger the camera.

        Returns
        -------
        dict
            Server reply.
        """
        msg = {
            "cmd": "new_run",
            "camera_params": camera_params,
            "data_filepath": data_filepath,
            "save_data": save_data,
            "N_img": N_img,
            "N_shots": N_shots,
            "N_pwa_per_shot": N_pwa_per_shot,
            "imaging_type": imaging_type,
            "run_id": run_id,
        }
        return self._send_recv(msg)

    def send_xvars(self, scan_xvars: list):
        """
        Send the current experiment xvar keys and values for a shot.

        Reads each xvar's current value from ``xvar.values[xvar.counter]``
        and forwards the resulting dict to all connected viewer clients.

        Parameters
        ----------
        scan_xvars : list of :class:`waxa.base.xvar.xvar`
            The scan variables for the current run.

        Returns
        -------
        dict
            Server reply (``{"cmd": "ack"}``).
        """
        xvars = {xv.key: xv.values[xv.counter] for xv in scan_xvars}
        self._send({"cmd": "xvars", "xvars": xvars})
        # return self._send_recv({"cmd": "xvars", "xvars": xvars})

    def send_run_complete(self):
        """
        Signal that the experiment run is complete.

        Returns
        -------
        dict
            Server reply (``{"cmd": "ack"}``).
        """
        return self._send_recv({"cmd": "run_complete"})

    def check_status(self):
        """
        Query the server's current status.

        Returns
        -------
        dict
            Status dict with keys ``camera_ready``, ``grab_active``, etc.
        """
        return self._send_recv({"cmd": "status"})
