import h5py, time
import numpy as np
import os

from waxa.data import DataSaver
from waxa.data.server_talk import server_talk as st
from waxa.config.timeouts import (DEFAULT_TIMEOUT, N_NOTIFY,
                                   CHECK_CAMERA_READY_ACK_PERIOD, REMOVE_DATA_POLL_INTERVAL,
                                   REMOVE_DATA_TIMEOUT,
                                   CHECK_FOR_DATA_AVAILABLE_PERIOD as CHECK_PERIOD)

def nothing():
    pass

class Scribe():
    def __init__(self, data_filepath="", server_talk=None):
        if server_talk == None:
            self.server_talk = st()
        else:
            self.server_talk = server_talk
        self.ds = DataSaver(server_talk=server_talk)
        if data_filepath != "":
            self.data_filepath = data_filepath

    def wait_for_data_available(self,openmode='r+',
                                check_period=CHECK_PERIOD,
                                timeout=DEFAULT_TIMEOUT,
                                check_interrupt_method=nothing,
                                require_data_group=True):
        """Blocks until the file at self.datapath is available and fully populated.

        reserve_run_id_and_path() now creates *and* populates the file inside a
        single exclusive open, so by the time the experiment knows the filepath
        the 'data' group exists.  The 'data' check below is therefore only
        defense-in-depth: hitting it means file creation did not complete, and
        we must time out rather than spin — an earlier version `continue`d past
        the deadline check and hung forever, silently discarding a whole run's
        images while the caller waited.

        Pass ``require_data_group=False`` when you only need the file to be
        openable (i.e. not held by another process), as when deleting it.
        """
        t0 = time.time()
        count = 0
        reason = "file busy"
        while True:
            # Deadline is checked at the top so that *every* retry path below
            # is bounded, including the 'data'-missing branch.
            if timeout > 0. and time.time() - t0 > timeout:
                raise ValueError(
                    f"Timed out waiting for data to be available after "
                    f"{time.time() - t0:.1f} s ({reason}): {self.data_filepath}"
                )
            try:
                if check_interrupt_method():
                    break
                f = h5py.File(self.data_filepath, openmode)
                # Guard against a file whose 'data' group was never written
                # (incomplete creation).
                if require_data_group and 'data' not in f:
                    f.close()
                    reason = "file has no 'data' group"
                    time.sleep(check_period)
                    continue
                return f
            except Exception as e:
                if "Unable to" in str(e) or "Invalid file name" in str(e) or "cannot access" in str(e):
                    # file is busy -- wait for available
                    reason = f"file busy ({e})"
                    count += 1
                    time.sleep(check_period)
                    if count == N_NOTIFY:
                        count = 0
                        print("Can't open data. Is another process using it?")
                else:
                    raise e
            self._check_data_file_exists()

    def wait_for_camera_ready(self,timeout=-1.) -> bool:
        # New path: delegate to the ZMQ client when available.
        if getattr(self, 'live_od_client', None) is not None:
            cam_timeout = timeout if timeout > 0. else 60.0
            self.live_od_client.wait_cam_ready(timeout=cam_timeout)
            print('Acknowledged camera ready signal.')
            return True

        # Legacy path: the old CameraMother file-watching mechanism has been
        # removed; camera_ready is no longer written to HDF5. Polling here
        # would block for the full timeout (45 s) and then crash. Fail fast
        # instead with a clear diagnostic.
        raise RuntimeError(
            "wait_for_camera_ready: no LiveOD server connection (live_od_client "
            "is not set) but setup_camera=True. The legacy HDF5-polling path is "
            "no longer supported (CameraMother file-watching was removed). Either "
            "ensure the LiveOD server window is running on the control PC, or "
            "pass setup_camera=False / suppress_live_od=True to Base.__init__."
        )
        # --- legacy code kept for reference (unreachable) ---
        count = 1
        t0 = time.time()
        waiting = True
        while waiting:
            try:
                self._check_data_file_exists()
            except:
                break
            if np.mod(count,N_NOTIFY) == 0:
                print('Waiting for camera ready.') 
                print(self.run_info.run_id)
            
            if timeout > 0.:
                if time.time() - t0 > timeout:
                    self.remove_incomplete_data()
                    raise ValueError("Waiting for camera ready timed out.")

            with self.wait_for_data_available() as f:
                if f.attrs['camera_ready']:
                    f.attrs['camera_ready_ack'] = 1
                    print('Acknowledged camera ready signal.')
                    waiting = False
                else:
                    count += 1
            time.sleep(CHECK_PERIOD)
        return True

    def mark_camera_ready(self,check_interrupt_method=nothing):
        with self.wait_for_data_available(check_interrupt_method=check_interrupt_method) as f:
            f.attrs['camera_ready'] = 1

    def check_camera_ready_ack(self,check_interrupt_method=nothing):
        while True:
            with self.wait_for_data_available(check_interrupt_method=check_interrupt_method) as f:
                if f.attrs['camera_ready_ack']:
                    print('Received ready acknowledgement.')
                    break
                else:
                    time.sleep(CHECK_CAMERA_READY_ACK_PERIOD)
        
    def write_data(self, expt_filepath):
        with self.wait_for_data_available() as f:
            self.ds.save_data(self, expt_filepath, f)
            print("Done!")

    def remove_incomplete_data(self,delete_data_bool=True):
        # msg = "Something went wrong."
        if delete_data_bool:
            # File may already have been deleted by the server (e.g. via
            # _finalize_reset_run) before CameraBaby's grab loop timed out
            # and called dishonorable_death.  If so, there is nothing to do.
            if not getattr(self, 'data_filepath', None) or not os.path.exists(self.data_filepath):
                return
            msg = "Destroying incomplete data."
            count = 0
            while True:
                try:
                    # The open is only here to confirm no one still holds the
                    # file, so don't require a 'data' group — a file that never
                    # got one is exactly what we are tearing down.  Bounded
                    # because this runs on the abort path.
                    with self.wait_for_data_available(
                            check_period=REMOVE_DATA_POLL_INTERVAL,
                            timeout=REMOVE_DATA_TIMEOUT,
                            require_data_group=False) as f:
                        pass
                    os.remove(self.data_filepath)
                    print(msg)
                except Exception as e:
                    if "Unable to" in str(e) or "Invalid file name" in str(e) or "cannot access" in str(e):
                        # file is busy -- wait for available
                        count += 1
                        time.sleep(CHECK_PERIOD)
                        if count == N_NOTIFY:
                            count = 0
                            print("Can't open data. Is another process using it?")
                    elif "Timed out waiting for data" in str(e):
                        # Can't open it (e.g. no 'data' group), but deletion may
                        # still succeed — that's all we actually want here.
                        try:
                            os.remove(self.data_filepath)
                            print(msg)
                        except Exception as rm_exc:
                            print(f"Could not delete incomplete data: {rm_exc}")
                            raise e
                    else:
                        raise e
                # Done as soon as the file is gone.  NB: _check_data_file_exists
                # is overridden in this class to report the *reset flag*, not
                # file existence, so on an aborted run it stays True and would
                # spin this loop on an already-deleted file.
                if not os.path.exists(self.data_filepath):
                    break
                if not self._check_data_file_exists(raise_error=False):
                    break

    # def _check_data_file_exists(self, raise_error=True) -> bool:
    #     """
    #     Checks if the data file exists if saving data is enabled. Raises an
    #     error if not found.
    #     """
    #     if hasattr(self, 'run_info'):
    #         filepath = getattr(self.run_info, 'filepath', None)
    #         if isinstance(filepath, list):
    #             # If filepath is a list, check all paths
    #             paths = filepath
    #         else:
    #             paths = [filepath]
    #         for path in paths:
    #             if isinstance(path, np.ndarray):
    #                 path = path.item() if path.size == 1 else None
    #             if path and not os.path.exists(path):
    #                 if raise_error:
    #                     if hasattr(self,'monitor'):
    #                         self.monitor.update_device_states()
    #                         self.monitor.signal_end()
    #                     raise RuntimeError(f"Data file for run ID {self.run_info.run_id} not found.")
    #                 else:
    #                     return False
    #         return True
        
    def _check_for_abort_signal(self, raise_error=True) -> bool:
        """Override: use ZMQ poll to check for a reset request instead of
        checking whether the data file still exists on disk.

        Falls back to the parent (file-based) check when no live_od_client
        is attached (e.g. suppress_live_od=True runs, or no liveOD server).
        """
        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            # Reuse the reset flag cached from the last SHOT_COMPLETE reply
            # instead of issuing a separate POLL round-trip every shot.  This
            # RPC runs from the scan @kernel, so removing the extra ZMQ
            # request-reply removes real latency from the real-time timeline.
            # A reset requested mid-shot is already caught by shot_complete()
            # (which raises TerminationRequested); the worst case here is a
            # one-shot delay in aborting.
            reset = getattr(_client, 'last_reset_requested', False)
            if reset and raise_error:
                if hasattr(self,'monitor'):
                    self.monitor.update_device_states()
                    self.monitor.signal_end()
                _client.abort_run()
                raise RuntimeError(f'Acquisition for run {self.run_info.run_id} aborted.')
            return reset
        else:
            return False
        
    def _check_data_file_exists(self, raise_error=True) -> bool:
        return self._check_for_abort_signal(raise_error)

    def _send_abort_to_server(self):
        """Notify the liveOD server that the run has been aborted due to
        an RTIOUnderflow.  Called as an RPC from the scan kernel after the
        current scan loop iteration completes.

        If ``run_info.save_on_underflow`` is set, pads any partially-captured
        scope data to the full shot count and returns normally so that
        ``post_scan()`` and ``analyze()`` execute as usual — ``end_wax()`` then
        handles ``cleanup_scanned()``, monitor cleanup, and sending the final
        END_RUN payload to the server.

        Otherwise (default), sends ABORT_RUN to the server and raises
        RuntimeError to prevent ``analyze()`` from executing.
        """
        save_on_underflow = bool(getattr(self.run_info, 'save_on_underflow', False))

        if save_on_underflow and self.run_info.save_data:
            # Pad scope data so reshape_data() sees the full (*xvardims,...) shape
            if hasattr(self, 'scope_data') and self.scope_data._scope_trace_taken:
                n_shots = int(getattr(self.params, 'N_shots_with_repeats', 1))
                self.scope_data.pad_to_n_shots(n_shots)
            print(f'[Scanner] RTIOUnderflow on run {self.run_info.run_id}: '
                  f'save_on_underflow=True — proceeding to analyze() to save partial data.')
            return  # let post_scan() -> run() -> analyze() -> end_wax() handle saving

        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            if hasattr(self, 'monitor'):
                self.monitor.update_device_states()
                self.monitor.signal_end()
            _client.abort_run()
        print(f'[Scanner] RTIOUnderflow: run {self.run_info.run_id} aborted.')
        raise RuntimeError(f'RTIOUnderflow: run {self.run_info.run_id} aborted.')