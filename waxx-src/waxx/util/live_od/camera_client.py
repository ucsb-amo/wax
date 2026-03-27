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

from artiq.language import TBool

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
                # Signal the server we're done and wait for acknowledgment
                reply = self._send_recv({"cmd": "disconnect"})
            except Exception as e:
                # If the send/recv fails, the connection is already broken
                print(f"Error sending disconnect: {e}")
            try:
                self._persistent_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._persistent_sock.close()
            except Exception as e:
                print(e)
            self._persistent_sock = None

    # ------------------------------------------------------------------
    #  Low-level binary helpers
    # ------------------------------------------------------------------

    def _send(self, msg):
        if self._persistent_sock is None:
            raise RuntimeError("CameraClient is not connected. Call connect() first.")
        send_msg(self._persistent_sock, msg)

    def _recv(self):
        if self._persistent_sock is None:
            raise RuntimeError("CameraClient is not connected. Call connect() first.")
        return recv_msg(self._persistent_sock)

    def _send_recv(self, msg):
        """Send *msg* and block until the server replies."""
        self._send(msg)
        return self._recv()

    # ------------------------------------------------------------------
    #  High-level run commands
    # ------------------------------------------------------------------

    def send_new_run(self, camera_params, data_filepath, save_data,
                     setup_camera, N_img, N_shots, N_pwa_per_shot, imaging_type, run_id,
                     data_spec=None, run_info_attrs=None,
                     params_attrs=None, camera_params_attrs=None,
                     available_data_fields=None):
        """
        Send camera params and run metadata to the server.

        Blocks until the server has connected the camera and started the
        grab loop. Returns ``{"cmd": "camera_ready"}`` on success —
        at that point the experiment can proceed to trigger the camera.

        If ``data_spec`` is provided the server will create the HDF5 data
        file itself, so the experiment process never needs to mount the
        data network drive.

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
            "setup_camera": setup_camera,
            "N_img": N_img,
            "N_shots": N_shots,
            "N_pwa_per_shot": N_pwa_per_shot,
            "imaging_type": imaging_type,
            "run_id": run_id,
            "data_spec": data_spec or {},
            "run_info_attrs": run_info_attrs or {},
            "params_attrs": params_attrs or {},
            "camera_params_attrs": camera_params_attrs or {},
            "available_data_fields": available_data_fields or [],
        }
        return self._send_recv(msg)

    def send_xvars(self, scan_xvars: list, data_fields: dict | None = None):
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
        self._send({"cmd": "xvars", "xvars": xvars, "data_fields": data_fields or {}})
        # return self._send_recv({"cmd": "xvars", "xvars": xvars})

    def send_shot_done(self):
        """
        Signal that a shot is complete (independent of image acquisition).

        This allows the server to count shots independently from when images
        are actually grabbed, which may have variable delays due to camera
        timing or processing.

        Returns
        -------
        dict
            Server reply (``{"cmd": "ack"}``).
        """
        return self._send_recv({"cmd": "shot_done"})

    def send_run_complete(self):
        """
        Signal that the experiment run is complete.

        Returns
        -------
        dict
            Server reply (``{"cmd": "ack"}``).
        """
        return self._send_recv({"cmd": "run_complete"})

    def send_write_data(self, data, params_attrs, sort_info,
                        scope_data_list, texts):
        """
        Send final run data to the server for writing into the HDF5 file.

        Must be called before ``send_run_complete`` while the persistent
        connection is still open.  The server waits for the grab loop to
        finish before writing, so this call blocks until the file write is
        acknowledged.

        Parameters
        ----------
        data : dict
            ``{key: np.ndarray}`` for each non-external DataContainer
            (already unshuffled on the experiment side).
        params_attrs : dict
            Serialized ExptParams key→value dict (post cleanup_scanned).
        sort_info : dict
            Sort/shuffle metadata so the server can unshuffle external
            datasets (images/timestamps) in-place.
        scope_data_list : list of dict
            ``[{"label": str, "t": ndarray, "v": ndarray}, ...]``.
        texts : dict
            Source-file text content keyed by
            ``expt / params / cooling / imaging / control``.
        """
        return self._send_recv({
            "cmd": "write_data",
            "data": data,
            "params_attrs": params_attrs,
            "sort_info": sort_info,
            "scope_data_list": scope_data_list,
            "texts": texts,
        })

    def check_interrupted(self) -> bool:
        """
        Ask the server whether a reset has been triggered for the current run.

        Used by the experiment scan loop instead of checking for the data
        file's physical existence on disk.

        Returns
        -------
        bool
        """
        reply = self._send_recv({"cmd": "check_interrupted"})
        return bool(reply.get("interrupted", False))

    def check_status(self):
        """
        Query the server's current status.

        Returns
        -------
        dict
            Status dict with keys ``camera_ready``, ``grab_active``, etc.
        """
        return self._send_recv({"cmd": "status"})
