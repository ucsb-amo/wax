import time
import vxi11
import struct
import numpy as np

TIMEBASE_VALUES = (
        200e-12, 500e-12, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9,
        500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
        1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
MEMORY_DEPTH_VALUES = (10000, 100000, 1000000, 10000000, 100000000,
                                    20000, 200000, 2000000, 20000000, 200000000)

class SiglentSDS2000XPlus(vxi11.Instrument):
    _name = "Siglent SDS2000X Plus"
    center_code = 127
    full_code = 256
    grid = 10

    def __init__(self, host, *args, **kwargs) -> None:
        super(SiglentSDS2000XPlus, self).__init__(host, *args, **kwargs)
        # idn = self.idn.split(',')
        # self.vendor = idn[0]
        # self.product = idn[1]
        # self.serial = idn[2]
        # self.firmware = idn[3]

    def read_sweep(self, src_channel:int, preamble=None):
        """_summary_

        :param src_channel: The channel to be sampled. Zero-indexed.
        """
        # while True:
        #     res = self.get_trigger_status()
        #     if res == SiglentSDSTriggerStatus.STOP.value:
        #         break

        # Send command that specifies the source waveform to be transferred

        self.write(":WAVeform:STARt 0")
        self.write(f":WAVeform:SOURce C{src_channel+1}")
        if preamble == None:
            preamble = self.get_waveform_preamble()
        adc_bit = preamble[-1]

        points = self.query(":ACQuire:POINts?").strip()
        points = float(self.query(":ACQuire:POINts?").strip())
        one_piece_num = float(self.query(":WAVeform:MAXPoint?").strip())
        if points > one_piece_num:
            self.write(":WAVeform:POINt {}".format(one_piece_num))
        if adc_bit > 8:
            self.write(":WAVeform:WIDTh WORD")

        read_times = int(np.ceil(points / one_piece_num))
        recv_all = []
        for i in range(0, read_times):
            start = i * one_piece_num
            self.write(":WAVeform:STARt {}".format(start))
            self.write("WAV:DATA?")
            recv_rtn = self.read_raw()
            block_start = recv_rtn.find(b'#')
            data_digit = int(recv_rtn[block_start + 1:block_start + 2])
            data_start = block_start + 2 + data_digit + 1
            recv = list(recv_rtn[data_start:-2:2])
            recv_all += recv

        try:
            v = self.convert_to_voltage(recv_all, preamble)
            t = self.waveform_time_axis(preamble)
            self._last_trace = np.array([t,v])
        except Exception as e:
            print(e)

        return self._last_trace
    
    def get_waveform_preamble(self):
        """The query returns the parameters of the source using by the command 
        :WAVeform:SOURce.

        Returns:
            tuple: (total_points, vdiv, voffset, code_per_div, timebase, delay, interval, delay)
                total_points (int): total number of waveform points
                vdiv (float): vertical scale in volts per division (already multiplied by probe)
                voffset (float): vertical offset in volts (already multiplied by probe)
                code_per_div (float): ADC code value per division (already multiplied by probe)
                timebase (int): timebase setting/index
                delay (float): horizontal delay (seconds)
                interval (float): sampling interval between points (seconds)
        """
        recv_all = self.query_raw(":WAVeform:PREamble?")
        recv = recv_all[recv_all.find(b'#') + 11:]

        # WAVE_ARRAY_1 = recv[0x3c:0x3f + 1]
        wave_array_count = recv[0x74:0x77 + 1]
        # first_point = recv[0x84:0x87 + 1]
        sp = recv[0x88:0x8b + 1]
        v_scale = recv[0x9c:0x9f + 1]
        v_offset = recv[0xa0:0xa3 + 1]
        interval = recv[0xb0:0xb3 + 1]
        code_per_div = recv[0xa4:0xa7 + 1]
        adc_bit = recv[0xac:0xad + 1]
        delay = recv[0xb4:0xbb + 1]
        tdiv = recv[0x144:0x145 + 1]
        probe = recv[0x148:0x14b + 1]

        # data_bytes = struct.unpack('i', WAVE_ARRAY_1)[0]
        point_num = struct.unpack('i', wave_array_count)[0]
        # fp = struct.unpack('i', first_point)[0]
        sp = struct.unpack('i', sp)[0]
        interval = struct.unpack('f', interval)[0]
        delay = struct.unpack('d', delay)[0]
        tdiv_index = struct.unpack('h', tdiv)[0]
        probe = struct.unpack('f', probe)[0]
        vdiv = struct.unpack('f', v_scale)[0] * probe
        offset = struct.unpack('f', v_offset)[0] * probe
        code = struct.unpack('f', code_per_div)[0]
        adc_bit = struct.unpack('h', adc_bit)[0]
        tdiv = TIMEBASE_VALUES[tdiv_index]

        return point_num, vdiv, offset, interval, delay, tdiv, code, adc_bit
    
    def convert_to_voltage(self, raw_array, preamble=None) -> np.ndarray:
        # Get the parameters of the source 
        if preamble == None:
            preamble = self.get_waveform_preamble()
        _, vdiv, ofst, _, _, _, vcode_per, adc_bit = preamble

        # handle >8 bit numbers if given by scope (adc_bit > 8)
        raw_array = np.array(raw_array)
        if adc_bit > 8:
            d = np.zeros(int(len(raw_array)/2))
            d = raw_array[1::2] * 256 + raw_array[::2]
        else:
            d = raw_array
        # handle int overflow
        mask = d > pow(2, adc_bit - 1) - 1
        d[mask] = d[mask] - pow(2, adc_bit)

        volt_value = np.array(d) / vcode_per * float(vdiv) - float(ofst)

        return volt_value
    
    def waveform_time_axis(self, preamble=None) -> np.ndarray:
        """Create time axis (seconds) from waveform preamble.

        Expects preamble as returned by get_waveform_preamble():
        (total_points, vdiv, voffset, code_per_div, timebase, delay, interval)
        """
        if preamble == None:
            preamble = self.get_waveform_preamble()
        point_num, _, _, interval, trdl, tdiv, _, _ = preamble
        return float(trdl) - (float(tdiv)*self.grid / 2) + np.arange(point_num) * interval
    
    def query(self, message, *args, **kwargs):
        """
        Write a message to the scope and read back the answer.
        See :py:meth:`vxi11.Instrument.ask()` for optional parameters.
        """
        return self.ask(message, *args, **kwargs)
    
    def query_raw(self, message, *args, **kwargs):
        """
        Write a message to the scope and read a (binary) answer.

        This is the slightly modified version of :py:meth:`vxi11.Instrument.ask_raw()`.
        It takes a command message string and returns the answer as bytes.

        :param str message: The SCPI command to send to the scope.
        :return: Data read from the device
        """
        data = message.encode('utf-8')
        return self.ask_raw(data, *args, **kwargs)
    
    @property
    def idn(self):
        """The command query identifies the instrument type and software version. The 
        response consists of four different fields providing information on the 
        manufacturer, the scope model, the serial number and the firmware revision.

        :return: Siglent Technologies,<model>,<serial_number>,<firmware>
        """
        return self.query("*IDN?")
    
    @property
    def timebase_scale(self) -> float:
        """The query returns the current horizontal scale setting in seconds per 
        division for the main window.

        :return: float

        """
        return float(self.query(":TIMebase:SCALe?"))
    
    @timebase_scale.setter
    def timebase_scale(self, new_timebase):
        """The command sets the horizontal scale per division for the main window.

        :param new_timebase: Value to set the horizontal timebase
        """
        self.write(":TIMebase:SCALe {}".format(new_timebase))

    @property
    def memory_depth(self) -> int:
        """The query returns the maximum memory depth.

        :return: int
                    Returns the maximum memory depth
        """
        return int(self.query(":ACQuire:MDEPth?"))
    
    @memory_depth.setter
    def memory_depth(self, mdepth: int):
        mdepth = min(MEMORY_DEPTH_VALUES, key=lambda x:abs(x-mdepth))
        self.write(":ACQuire:MDEPth {}".format(mdepth))
    
    def get_trigger_status(self):
        """The command query returns the current state of the trigger.

        :return: str
                    Returns either "Arm", "Ready", "Auto", "Trig'd", "Stop", "Roll"
        """
        return self.query(":TRIGger:STATus?")
    
    def autosetup(self):
        """ This command attempts to automatically adjust the trigger, vertical, and 
        horizontal controls of the oscilloscope to deliver a usable display of the 
        input signal. Autoset is not recommended for use on low frequency events 
        (< 100 Hz).

        :return: Nothing
        """
        self.write(":AUToset")

    def set_trigger_run(self):
        """The command sets the oscilloscope to run
        """
        self.write(":TRIGger:RUN")
    
    def set_single_trigger(self):
        """The command sets the mode of the trigger.

        The backlight of SINGLE key lights up, the oscilloscope enters the 
        waiting trigger state and begins to search for the trigger signal that meets 
        the conditions. If the trigger signal is satisfied, the running state shows 
        Trig'd, and the interface shows stable waveform. Then, the oscilloscope stops 
        scanning, the RUN/STOP key becomes red, and the running status shows Stop. 
        Otherwise, the running state shows Ready, and the interface does not display 
        the waveform.

        :return: Nothing
        """
        self.write(":TRIGger:MODE SINGle")
    
    def set_normal_trigger(self):
        """The command sets the mode of the trigger.
        
        The oscilloscope enters the wait trigger state and begins to search for 
        trigger signals that meet the conditions. If the trigger signal is satisfied, 
        the running state shows Trig'd, and the interface shows stable waveform. 
        Otherwise, the running state shows Ready, and the interface displays the last 
        triggered waveform (previous trigger) or does not display the waveform (no 
        previous trigger).

        :return: Nothing
        """
        self.write(":TRIGger:MODE NORMal")
    
    def set_auto_trigger(self):
        """The command sets the mode of the trigger.
        
        The oscilloscope begins to search for the trigger signal that meets the 
        conditions. If the trigger signal is satisfied, the running state on the top 
        left corner of the user interface shows Trig'd, and the interface shows stable 
        waveform. Otherwise, the running state always shows Auto, and the interface 
        shows unstable waveform.

        :return: Nothing
        """
        self.write(":TRIGger:MODE AUTO")
    
    def set_force_trigger(self):
        """The command sets the mode of the trigger.
        
        Force to acquire a frame regardless of whether the input signal meets the 
        trigger conditions or not.

        :return: Nothing
        """
        self.write(":TRIGger:MODE FTRIG")
    
    def get_trigger_mode(self):
        """The query returns the current mode of trigger.

        :return: str
                    Returns either "SINGle", "NORMal", "AUTO", "FTRIG"
        """
        return self.query(":TRIGger:MODE?")
    
    def set_rising_edge_trigger(self):
        """The command sets the slope of the slope trigger to Rising Edge

        :return: Nothing
        """
        self.write(":TRIGger:SLOPe:SLOPe RISing")

    def set_falling_edge_trigger(self):
        """The command sets the slope of the slope trigger to Falling Edge

        :return: Nothing
        """
        self.write(":TRIGger:SLOPe:SLOPe FALLing")

    def set_alternate_edge_trigger(self):
        """The command sets the slope of the slope trigger to Falling Edge

        :return: Nothing
        """
        self.write(":TRIGger:SLOPe:SLOPe ALTernate")

    def get_edge_trigger(self):
        """The query returns the current slope of the slope trigger

        :return: str
                    Returns either "RISing", "FALLing", "ALTernate"
        """
        return self.query(":TRIGger:SLOPe:SLOPe?")

    def set_trigger_source(self, trig_channel:int):
        """The query returns the current trigger source of the slope trigger

        :param trig_channel: Trigger source
        """
        self.write(f":TRIGger:SLOPe:SOURce C{trig_channel}")

    def set_trigger_edge_level(self, level : float):
        """The command sets the trigger level of the edge trigger

        :param level: Trigger level
        """

        """
        TODO: trigger level needs to be between:
        [-4.1*vertical_scale-vertical_offset, 4.1*vertical_scale-vertical_offset]
        """
        self.write(":TRIGger:EDGE:LEVel {}".format(str(level)))

    def save_setup(self, file_location : str):
        """This command saves the current settings to internal or external memory 
        locations.

        Users can recall from local,net storage or U-disk according to requirements

        :param file_location: string of path with an extension “.xml”. 
        """
        if file_location.endswith(".xml"):
            self.write(':SAVE:SETup EXTernal,”{}”'.format(file_location))
        else:
            raise ValueError("Add in string that contains .xml")


    def recall_setup(self, file_location : str):
        """This command will recall the saved settings file from external sources.
        
        Users can recall from local,net storage or U-disk according to requirements

        :param file_location: string of path with an extension “.xml”. 
        """
        if file_location.endswith(".xml"):
            self.write(':RECall:SETup EXTernal,”{}”'.format(file_location))
        else:
            raise ValueError("Add in string that contains .xml")


    def channel_visibile(self, channel : int, visible : bool = True):
        """The command is used to whether display the waveform of the specified 
        channel or not.

        :param channel: 0 to (# analog channels) - 1
        :param visible: Diplay state, defaults to True
        """
        channel += 1
        assert 1 <= channel <= 4    # probably need to change to specific 
                                    # oscope

        visible = "ON" if visible else "OFF"

        self.write(":CHANnel{}:VISible {}".format(str(channel), visible))

    def is_channel_visible(self, channel : int):
        """The query returns whether the waveform display function of the selected 
        channel is on or off.

        :param channel: 0 to (# analog channels) - 1
        :return: bool
        """
        channel += 1
        assert 1 <= channel <= 4    # probably need to change to specific 
                                    # oscope

        resp = self.query("CHAN{}:VIS?".format(str(channel)))
                          
        return ( True if resp == "ON" else False )
        
    # def set_waveform_format_width(self, waveform_width : SiglentWaveformWidth):
    #     """The command sets the current output format for the transfer of waveform
    #     data.

    #     :param waveform_width:  SiglentWaveformWidth.BYTE or SiglentWaveformWidth.WORD
    #     """
    #     assert isinstance(waveform_width, SiglentWaveformWidth)

    #     self.write(":WAVeform:WIDTh {}".format(waveform_width.value))

    # def get_waveform_format_width(self) -> SiglentWaveformWidth:
    #     """The query returns the current output format for the transfer of waveform 
    #     data.
    #     """
    #     resp = self.query(":WAVeform:WIDTh?")

    #     match resp:
    #         case "BYTE":
    #             return SiglentWaveformWidth.BYTE
    #         case "WORD":
    #             return SiglentWaveformWidth.WORD
            
    # def calculate_voltage(self, x, vdiv, voffset, code_per_div):
    #     if x > self.center_code:
    #         x -= self.full_code

    #     return x * (vdiv/code_per_div) - voffset

    def arm(self):
        """Sets up the trigger signal to single
        """
        
        self.set_single_trigger()
        self.set_trigger_run()
        self.query("*OPC?")
            
    def default_setup(self):
        pass