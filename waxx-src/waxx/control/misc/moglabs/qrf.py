"""Class for the moglabs QRF."""

from typing import Literal, Optional, Union

from .mogdevice import MOGDevice


class QRF(MOGDevice):
    """
    Represents a moglabs QRF device.

    Parameters
    ----------
    addr : str
        The IP address of the device. Port can be specified with the format "addr:port".
    port : int, optional
        The port number of the device. Default is 7802.
    timeout : int, optional
        The timeout for the connection in seconds. Default is 1.
    """

    def __init__(self, addr: str, port: int = 7802, timeout=1):
        # do not check connection since `self.info` has been redefined in this class
        super().__init__(addr=addr, port=port, timeout=timeout, check=False)
        self.channels = {ch: Channel(self, ch) for ch in range(1, 5)}

    def cmd(self, cmd: str):
        resp = self.ask(cmd)
        if resp.startswith("OK"):
            return resp
        else:
            if cmd.startswith("TABLE,ENTRY"):
                # this command doesn't send an 'OK'
                if resp.startswith("Invalid"):
                    raise RuntimeError(resp)
                else:
                    return resp
            raise RuntimeError(resp)

    def reboot(self) -> None:
        """
        Initiate microcontroller reset, causing unit to reboot. Note that all
        communications links will be immediately closed.
        """
        self.cmd("REBOOT")

    @property
    def info(self) -> str:
        """Report information about the unit."""
        return self.ask("INFO")

    @property
    def version(self) -> str:
        """Report versions of firmware currently running on device."""
        return self.ask("VERSION")

    @property
    def temperature(self) -> float:
        """Report measured temperatures and fan speeds."""
        return float(self.ask("TEMP").split(" ")[0])

    def sleep(self, dt: int) -> None:
        """
        Pause microcontroller operation for dt milliseconds.

        Intended for use in simple scripts to wait for a short amount of time, e.g. for
        tables to finish execution. Note that the microcontroller will be unresponsive
        during this time.

        Parameters
        ----------
        dt : int
            Duration of sleep in milliseconds.
        """
        _ = self.cmd(f"SLEEP,{dt}")

    @property
    def clock_source(self) -> str:
        """Report the current clock source and frequency."""
        return self.ask("CLKSRC")

    @clock_source.setter
    def clock_source(self, clk: Literal["INT", 25, 50, 100, 125, 500]):
        """Set or query the clock source and frequency."""
        if clk not in ["INT", 25, 50, 100, 125, 500]:
            raise ValueError("clk must be one of 'INT', 25, 50, 100, 125, 500")
        if clk == "INT":
            return self.cmd("CLKSRC,INT")
        else:
            return self.cmd(f"CLKSRC,EXT,{clk}")

    def copy_table(self, src: int, dest: int) -> None:
        """
        Copy the table data from the src channel to the dest channel.

        Parameters
        ----------
        src, dest : int
            Source and destination channel.
        """
        _ = self.cmd(f"TABLE,APPEND,{src},{dest}")

    def align_phase(self, ch: Optional[list[int]] = None) -> None:
        """
        Simultaneously resets the phase accumulators of the associated channels.

        This may be necessary to re-establish a stable phase relationship between two
        channels after their frequencies are adjusted.

        Parameters
        ----------
        ch : list of int
            The channels to phase-align. If no channels are specified, all channels are
            simultaneously reset.
        """
        cmd = "ALIGNPH"
        if ch is not None:
            for c in ch:
                cmd += f",{c}"
        _ = self.cmd(cmd)

    def start(self, ch: Optional[Union[list[int], int]] = None):
        """
        Provide a software trigger to start table execution on the specified channel(s).

        Parameters
        ----------
        ch : int or list of int
            The channel(s) that should be started.
        """
        cmd = "TABLE,START"
        if ch is not None:
            if isinstance(ch, int):
                ch = [ch]
            for c in ch:
                cmd += f",{c}"
        return self.cmd(cmd)

    def stop(self, ch: Optional[Union[list[int], int]] = None):
        """
        Stop table execution at the end of the current step for the specified
        channel(s).

        Parameters
        ----------
        ch : int or list of int
            The channel(s) that should be stopped.
        """
        cmd = "TABLE,STOP"
        if ch is not None:
            if isinstance(ch, int):
                ch = [ch]
            for c in ch:
                cmd += f",{c}"
        return self.cmd(cmd)


class Channel:
    def __init__(self, qrf: QRF, ch: int) -> None:
        self.qrf = qrf
        self.ch = ch

    @property
    def status(self) -> int:
        """The current operational status of the specified channel."""
        return int(self.qrf.ask(f"STATUS,{self.ch}"))

    def set_mode(self, mode: Literal["NSB", "NSA", "TSB"]):
        """Set the operational mode ("NSB", "NSA", "TSB") of the given channel."""
        self.qrf.cmd(f"MODE,{self.ch},{mode}")

    def turn_on(self, mode: Literal["SIG", "POW", "ALL"] = "ALL"):
        """
        Enable the signal ("SIG"), amplifier ("POW") or both ("ALL", default) outputs.
        """
        self.qrf.cmd(f"ON,{self.ch},{mode}")

    def turn_off(self, mode: Literal["SIG", "POW", "ALL"] = "ALL"):
        """
        Disable the signal ("SIG"), amplifier ("POW") or both ("ALL", default) outputs.
        """
        self.qrf.cmd(f"OFF,{self.ch},{mode}")

    @property
    def frequency(self) -> float:
        """Frequency of the RF output in MHz."""
        resp = self.qrf.ask(f"FREQ,{self.ch}")
        return float(resp.split("MHz")[0])

    @frequency.setter
    def frequency(self, value: float) -> None:
        _ = self.qrf.cmd(f"FREQ,{self.ch},{value}")

    @property
    def power(self) -> float:
        """Output power in dBm."""
        resp = self.qrf.ask(f"POW,{self.ch}")
        return float(resp.split("dBm")[0])

    @power.setter
    def power(self, value: float) -> None:
        _ = self.qrf.cmd(f"POW,{self.ch},{value}")

    @property
    def limit(self) -> float:
        """
        Maximum RF power for the channel in dBm. Output power is reduced to the limit
        if it is set below the current power level.
        """
        resp = self.qrf.ask(f"LIMIT,{self.ch}")
        return float(resp.split("dBm")[0])

    @limit.setter
    def limit(self, value: Union[float, str]):
        _ = self.qrf.cmd(f"LIMIT,{self.ch},{value}")

    @property
    def phase(self) -> float:
        """Phase offset of the RF output in degrees."""
        resp = self.qrf.ask(f"PHASE,{self.ch}")
        return float(resp.split("deg")[0])

    @phase.setter
    def phase(self, value: float) -> None:
        _ = self.qrf.cmd(f"PHASE,{self.ch},{value}")

    # Table mode commands
    @property
    def table_status(self) -> str:
        """The current execution status of the table."""
        return self.qrf.ask(f"TABLE,STATUS,{self.ch}")

    def get_table_entry(self, num: int) -> str:
        """
        Get the current entry of the table.

        Parameters
        ----------
        num : int
            The entry to get (1 to 8191).
        """
        return self.qrf.ask(f"TABLE,ENTRY,{self.ch},{num}")

    def set_table_entry(
        self,
        num: int,
        freq: float,
        power: float,
        phase: float = 0.0,
        dur: int = 0,
        flags: Optional[
            Union[Literal["SIG", "POW", "TRIG"], list[Literal["SIG", "POW", "TRIG"]]]
        ] = None,
    ) -> None:
        """
        Set or edit the specified table entry.

        Parameters
        ----------
        num : int
            The entry to edit (1 to 8191).
        freq : float
            Frequency to output during this step in MHz.
        power : float
            Output power during this step in dBm.
        phas : float
            Phase offset of the RF output for this step in deg.
        dur : float or str
            Duration of this step in multiple of 5 us If 0, the entry is held until a
            hardware trigger is received.
        flags : str or list of str
            One or a list of the following flags:
            - "SIG": Switch off the RF signal for this step, disabling the output. Must
                be repeated in subsequent steps for the signal to remain off.
            - "POW": Switch off the RF power amplifier for this step, disabling the
                output. Must be repeated in subsequent steps for the signal to remain
                off.
            - "TRIG": Equivalent to setting duration to 0; that is, wait for external
                trigger.
        """
        cmd = f"TABLE,ENTRY,{self.ch},{num},{freq},{power},{phase}{dur}"
        if flags:
            cmd += f",{flags}"
        _ = self.qrf.cmd(cmd)

    def append_table_entry(
        self,
        freq: float,
        power: float,
        phase: float = 0.0,
        dur: int = 0,
        flags: Optional[
            Union[Literal["SIG", "POW", "TRIG"], list[Literal["SIG", "POW", "TRIG"]]]
        ] = None,
    ):
        """
        Insert the specified entry at the end of the table and increment entry counter.

        Parameters
        ----------
        freq : float
            Frequency to output during this step in MHz.
        power : float
            Output power during this step in dBm.
        phas : float
            Phase offset of the RF output for this step in deg.
        dur : float or str
            Duration of this step in multiple of 5 us If 0, the entry is held until a
            hardware trigger is received.
        flags : str or list of str
            One or a list of the following flags:
            - "SIG": Switch off the RF signal for this step, disabling the output. Must
                be repeated in subsequent steps for the signal to remain off.
            - "POW": Switch off the RF power amplifier for this step, disabling the
                output. Must be repeated in subsequent steps for the signal to remain
                off.
            - "TRIG": Equivalent to setting duration to 0; that is, wait for external
                trigger.
        """
        cmd = f"TABLE,APPEND,{self.ch},{freq},{power},{phase},{dur}"
        if flags:
            cmd += f",{flags}"
        _ = self.qrf.cmd(cmd)

    def get_table_entry_as_hex(self, num: int) -> str:
        """
        Get hexadecimal representation of a table entry, returning the internal
        hexadecimal representation of the associated frequency, amplitude and phase.

        Parameters
        ----------
        num : int
            The entry to edit (1 to 8191).
        """
        return self.qrf.ask(f"TABLE,HEXENTRY,{self.ch},{num}")

    @property
    def table_length(self) -> int:
        return int(self.qrf.ask(f"TABLE,ENTRIES,{self.ch}"))

    @table_length.setter
    def table_length(self, num: int) -> None:
        """
        The last table entry number. Incorrectly setting the number of entries can
        result in undefined behaviour.

        Parameters
        ----------
        num : int (optional)
            The entry to edit (1 to 8191).
        """
        cmd = f"TABLE,ENTRIES,{self.ch},{num}"
        _ = self.qrf.cmd(cmd)

    def clear_table(self):
        """Stops and clears the table entries from the specified channel."""
        self.qrf.cmd(f"TABLE,CLEAR,{self.ch}")

    @property
    def table_name(self) -> str:
        """The name of the table for identification purposes."""
        return self.qrf.ask(f"TABLE,NAME,{self.ch}")

    @table_name.setter
    def table_name(self, name: str):
        """Assign a character string to the table for identification purposes."""
        cmd = f"TABLE,NAME,{self.ch},{name}"
        _ = self.qrf.cmd(cmd)

    def arm(self):
        """
        Load the table for execution and ready the output. The table then begins
        execution upon receiving a software or hardware trigger.
        """
        return self.qrf.cmd(f"TABLE,ARM,{self.ch}")

    def start(self) -> None:
        """Provide a software trigger to initiate table execution."""
        return self.qrf.cmd(f"TABLE,START,{self.ch}")

    def stop(self):
        """Terminate an executing table at the end of the current step."""
        _ = self.qrf.cmd(f"TABLE,STOP,{self.ch}")

    @property
    def rearm_enabled(self) -> bool:
        """
        Enable/disable the automatic re-arming (loading) of the table upon completion
        such that it can be started again from a hardware or software trigger.
        """
        return self.qrf.ask(f"TABLE,REARM,{self.ch}") == "on"

    @rearm_enabled.setter
    def rearm_enabled(self, enable: bool) -> None:
        _ = self.qrf.cmd(f"TABLE,REARM,{self.ch},{'on' if enable else 'off'}")

    @property
    def restart_enabled(self) -> bool:
        return self.qrf.ask(f"TABLE,RESTART,{self.ch}") == "on"

    @restart_enabled.setter
    def restart_enabled(self, enable: bool) -> None:
        """
        Enable/disable an automatic software-controlled restart of the table upon
        completion.
        """
        restart = "on" if enable else "off"
        _ = self.qrf.cmd(f"TABLE,RESTART,{self.ch},{restart}")

    def time_sync(self) -> None:
        """
        Synchronise internal clock to the external trigger on the specified
        channel. The first trigger will reset the timebase and a subsequent
        trigger will initiate table execution.
        """
        _ = self.qrf.cmd(f"TABLE,TRIGSYNC,{self.ch}")

    @property
    def edge(self):
        """TTL trigger edge for the table ("RISING" or "FALLING")."""
        return self.qrf.ask(f"TABLE,EDGE,{self.ch}")

    @edge.setter
    def edge(self, edge: Literal["RISING", "FALLING"]) -> None:
        if edge not in ["RISING", "FALLING"]:
            raise ValueError("edge must be 'RISING' or 'FALLING'")
        _ = self.qrf.cmd(f"TABLE,EDGE,{self.ch},{edge}")
