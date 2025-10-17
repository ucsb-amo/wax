from wax.control.artiq.DAC_CH import DAC_CH
from wax.control.artiq.TTL import TTL
from wax.control.artiq.DDS import DDS
from wax.config.expt_params import ExptParams
from wax.control.tweezer import tweezer_xmesh as mesh, TweezerMovesLib

from artiq.language.core import now_mu
from artiq.coredevice.core import Core
from artiq.experiment import rpc, kernel, delay, parallel, TFloat, portable, TArray, TInt32

import spcm
from spcm import units

import numpy as np

# di = 666420695318008 #causes failure
di = 0
dv = -1000.
dv_list = np.linspace(0.,1.,5)
dv_array = np.array([dv])
db_array = np.array([None])
T_AWG_RAMP_WRITE_DELAY = 100.e-3
T_AWG_RPC_DELAY = 25.e-3

VAL_TYPE_FREQ = 0
VAL_TYPE_AMP = 1

class TweezerTrap():
    def __init__(self,
                 position=dv,
                 amplitude=dv,
                 cateye:bool=False,
                 frequency=dv,
                 tweezer_xmesh=mesh(),
                 awg_trigger_ttl=TTL,
                 expt_params=ExptParams(),
                 core=Core):
        
        self.mesh = tweezer_xmesh
        self.moves = TweezerMovesLib()

        self.position = position
        self.amplitude = amplitude
        self.cateye = cateye
        if frequency != dv:
            self.frequency = frequency
            self.position = self.f_to_x(frequency)
        else:
            self.frequency = self.x_to_f(position)

        self.awg_trig_ttl = awg_trigger_ttl
        self.p = expt_params
        self.core = core
        self.dds = spcm.DDSCommandList
        self.dds: spcm.DDSCommandList

        if not hasattr(self.p,'idx_tweezer'):
            self.p.idx_tweezer = 0
        self.dds_idx = self.p.idx_tweezer
        self.p.idx_tweezer += 1

        if cateye:
            self.x_per_f = self.mesh.x_per_f_ce
        else:
            self.x_per_f = self.mesh.x_per_f_nce

        self.dummy_out = np.array([0.])
        self._value_final = 0.
        self.values = np.zeros((1000000,),dtype=float)
        self._N = 0
        self._N_steps_per_dds_write = 1000

    def update_x_rpc(self,x=dv,from_frequency=False) -> TFloat:
        """Updates the position attribute of the tweezer on the host device.

        Args:
            x (float): The new position (in m).

        Returns:
            TFloat: the new position (in m) of the tweezer trap.
        """        
        if from_frequency:
            self.position = self.f_to_x(self.frequency)
        else:
            if x != dv:
                self.position = x
        return self.position
    
    def update_amp_rpc(self,amp) -> TFloat:
        """Updates the position attribute of the tweezer on the host device.

        Args:
            amp (float): The new amplitude (from 0 to 1).

        Returns:
            TFloat: The new amplitude of the tweezer trap.
        """        
        self.amplitude = amp
        return self.amplitude
    
    def update_f_rpc(self,frequency=dv,from_position=True) -> TFloat:
        if from_position:
            self.frequency = self.x_to_f(self.position)
        else:
            if frequency != dv:
                self.frequency = frequency
        return self.frequency
    
    @kernel
    def update_f(self,f,update_x=True):
        self.frequency = self.update_f_rpc(f,from_position=False)
        if update_x:
            self.position = self.update_x_rpc()

    @kernel
    def update_x(self,x,update_freq=True):
        """Updates the position attribute of the tweezer on the core device.

        Args:
            x (float): The new position (in m).
        """
        self.position = self.update_x_rpc(x)
        if update_freq:
            self.frequency = self.update_f_rpc()

    @kernel
    def update_amp(self,amp):
        self.amplitude = self.update_amp_rpc(amp)

    def set_amp_rpc(self,amp):
        self.dds.amp(self.dds_idx,amp)
        self.dds.exec_at_trg()
        self.dds.write()

    @kernel
    def set_amp(self,amp,trigger=True):

        self.core.wait_until_mu(now_mu())
        self.set_amp_rpc(amp)
        self.update_amp(amp)
        delay(T_AWG_RPC_DELAY)
        if trigger:
            self.awg_trig_ttl.pulse(1.e-6)

    def set_position_rpc(self,x):
        self.dds.freq(self.dds_idx,self.x_to_f(x))
        self.dds.exec_at_trg()
        self.dds.write()

    @kernel
    def set_position(self,x,trigger=True):

        self.core.wait_until_mu(now_mu())
        self.set_position_rpc(x)
        self.update_x(x)
        delay(T_AWG_RPC_DELAY)
        if trigger:
            self.awg_trig_ttl.pulse(1.e-6)

    def set_frequency_rpc(self,f):
        self.dds.freq(self.dds_idx,f)
        self.dds.exec_at_trg()
        self.dds.write()

    @kernel
    def set_frequency(self,f,trigger=True):

        self.core.wait_until_mu(now_mu())
        self.set_frequency_rpc(f)
        self.update_f(f,update_x=True)
        delay(T_AWG_RPC_DELAY)
        if trigger:
            self.awg_trig_ttl.pulse(1.e-6)
    
    def compute_cubic_move(self,t_move,x_move,dt=dv):
        """Compute the frequency slopes required for a cubic move profile (zero
        intial and final velocity, displacement x_move in time t_move).

        Args:
            t_move (float): the total duration (in s) of the move.
            x_move (float): the total displacement for the move.

        Returns:
            TArray(TFloat): the frequency slopes for the move.
        """        
        if dt == dv:
            self.p.t_tweezer_movement_dt
        self.compute_slopes(t_move,
                            self.moves.cubic_move,
                            t_move,x_move,
                            dt = dt)
    
    def compute_sinusoidal_modulation(self,
                                      t_move,x_amplitude,
                                      modulation_frequency,
                                      t_mod_amp_ramp,
                                      dt=dv):
        """Compute the frequency slopes required for a sinusoidal move profile.

        Args:
            t_move (float): the total duration (in s) of the move.
            x_amplitude (float): the displacement amplitude (in m) for the move.
            modulation_frequency (float): the modulation frequency (in Hz) for
            the move.

        Returns:
            TArray(TFloat): the frequency slopes for the move.
        """        
        if dt == dv:
            self.p.t_tweezer_movement_dt
        self.compute_slopes(t_move,self.moves.sinusoidal_modulation,
                                    x_amplitude,
                                    modulation_frequency,
                                    t_mod_amp_ramp,
                                    dt = dt)
    
    
    def compute_linear_amplitude_ramp(self,t_ramp,amp_f,dt=dv,slopes=False):
        if dt == dv:
            self.p.t_tweezer_amp_ramp_dt
        self.compute_values(t_ramp,self.moves.linear,
                                    t_ramp,self.amplitude,amp_f,
                                    dt = dt,
                                    ramp_type=VAL_TYPE_AMP)

    @kernel
    def cubic_move(self,t_move,x_move,
                   dt=dv,trigger=True):
        """Executes a cubic move for this tweezer trap.

        Uses a move step time of dt = ExptParams.t_tweezer_movement_dt.

        Args:
            t_move (float): the total duration (in s) of the move.
            x_move (float): the total displacement for the move.
            trigger (bool): whether or not to trigger the move start.
        """
        if dt == dv:
            self.p.t_tweezer_movement_dt
        self.compute_cubic_move(t_move,x_move)
        self.move(t_move, trigger=trigger, slopes=True)
    
    @kernel
    def sine_move(self,t_mod,x_mod,f_mod,t_xmod_ramp=0.,
                  dt=dv,trigger=True):
        """Executes a sinusoidal move for this tweezer trap.

        Args:
            t_move (float): the total duration (in s) of the move.
            x_mod (float): the displacement amplitude (in m) for the move.
            f_mod (float): the modulation frequency (in Hz) for
            the move.
            t_xmod_ramp (float): if nonzero, the time (in s) to linearly ramp
            the modulation amplitude from 0 to x_mod.
        """
        if dt == dv:
            self.p.t_tweezer_movement_dt
        self.compute_sinusoidal_modulation(t_mod,x_mod,f_mod,t_xmod_ramp)
        self.move(t_mod, trigger=trigger, slopes=True)

    @kernel
    def linear_amplitude_ramp(self,t_ramp,amp_f,
                              dt=dv,trigger=True):
        if dt == dv:
            self.p.t_tweezer_amp_ramp_dt
        self.compute_linear_amplitude_ramp(t_ramp,amp_f)
        self.amp_ramp(t_ramp, amp_final=amp_f, trigger=trigger, slopes=False)
        
    @portable
    def x_to_f(self,x) -> TFloat:
        """Converts the given tweezer position x to the required AOD frequency.

        Args:
            x (float): Position (in m).

        Returns:
            TFloat: the AOD frequency (in Hz) corresponding to the given position x.
        """        
        self.dummy_out = self.mesh.x_to_f(x,self.cateye)
        return self.dummy_out[0]
        
    @portable
    def f_to_x(self,f)  -> TFloat:
        """Converts the given AOD frequency to the corresponding tweezer
        position.

        Args:
            f (float): AOD frequency (in Hz).

        Returns:
            TFloat: the position x (in m) corresponding to the given AOD
            frequency (in Hz).
        """        
        self.dummy_out = self.mesh.f_to_x(f)
        return self.dummy_out[0]

    def compute_slopes(self,t_move,
               x_vs_t_func,
               *x_vs_t_params,
               dt = dv,
               ramp_type=VAL_TYPE_FREQ):
        """Compute the frequency slopes required to implement the specified move
        profile x(t) from t=0 to t=t_move. 
        
        Uses a move step time of dt = ExptParams.t_tweezer_movement_dt.

        Args:
            t_move (float): the total duration (in s) of the move.
            x_vs_t_func (function): x(t) for the desired move. Should take an
            array of times (seconds) as its first argument, and then any number
            of parameter arguments.
            x_vs_t_params: any number of parameter arguments to be passed to
            x_vs_t_func as x_vs_t_func(t,*x_vs_t_params).

        Returns:
            TArray(TFloat): the frequency slopes for the move.
        """
        if dt == dv:
            dt = self.p.t_tweezer_movement_dt

        tarray = np.arange(0.,t_move,dt)
        self._N = len(tarray)
        self.values[0:(self._N-1)] = np.diff(x_vs_t_func(tarray,*x_vs_t_params)) / dt 
        self.values[self._N-1] = 0.0
        
        if ramp_type == VAL_TYPE_FREQ:
            self.values = self.values / self.x_per_f
            slope_min = self.dds.avail_freq_slope_step()
        elif ramp_type == VAL_TYPE_AMP:
            slope_min = self.dds.avail_amp_slope_step() * 1.00001

        self.values = np.where(
            (np.abs(self.values) < slope_min) & (self.values != 0),
               np.sign(self.values) * slope_min,
               self.values)
        
        self._value_final = np.sum( self.values * dt )
        
    def compute_values(self,t_move,
                       x_vs_t_func,
                       *x_vs_t_params,
                       dt = dv,
                       ramp_type=VAL_TYPE_FREQ):
        if dt == dv:
            dt = self.p.t_tweezer_amp_ramp_dt

        tarray = np.arange(0.,t_move,dt)
        self._N = len(tarray)
        self.values[0:self._N] = x_vs_t_func(tarray,*x_vs_t_params)

        if ramp_type == VAL_TYPE_FREQ:
            self.values = self.values / self.x_per_f
        elif ramp_type == VAL_TYPE_AMP:
            pass

        self._value_final = self.values[self._N-1]

    @kernel
    def move(self,
             t_move,
             dt = dv,
             trigger = True,
             slopes = True):
        """Sets the timeline cursor to the current RTIO time (wall-clock), then
        starts writing the slopes list to the awg.

        Args:
            compute_move_output (tuple of (float,ndarray)): A tuple containing
            the time of the move and the move's frequency slopes.
        """
        if dt == dv:
            dt = self.p.t_tweezer_amp_ramp_dt

        if slopes:
            x_final = self.position + self._value_final
        else:
            x_final = self._value_final

        self.core.wait_until_mu(now_mu())
        self.update_x(x_final)
        self.write_move(dt,slopes)
        delay(T_AWG_RPC_DELAY)

        if trigger:
            self.awg_trig_ttl.pulse(1.e-6)
            delay(t_move)

    @kernel
    def amp_ramp(self,
                 t_move,
                 amp_final,
                 dt = dv,
                 trigger = True,
                 slopes = False):
        """Sets the timeline cursor to the current RTIO time (wall-clock), then
        starts writing the amplitude slopes list to the awg.

        Args:
            compute_move_output (tuple of (float,ndarray)): A tuple containing
            the time of the move and the ramp's amplitude slopes.
        """
        if dt == dv:
            dt = self.p.t_tweezer_amp_ramp_dt

        if slopes:
            amp_final = self.amplitude + self._value_final
        else:
            amp_final = self._value_final

        self.core.wait_until_mu(now_mu())
        self.update_amp(amp_final)
        self.write_amp_ramp(dt,slopes)
        delay(T_AWG_RPC_DELAY)

        if trigger:
            self.awg_trig_ttl.pulse(1.e-6)
            delay(t_move)

    @rpc(flags={"async"})
    def write_move(self,dt=dv,slopes=True):
        """Writes the slopes list to the AWG at update interval dt.

        Args:
            slopes (ndarray): The list of frequency slopes (Hz/s) to be written
            to the awg.
        """   
        if dt == dv:
            dt = self.p.t_tweezer_movement_dt

        self.dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        self.dds.trg_timer(dt)
        self.dds.exec_at_trg()
        self.dds.write()

        i = 0

        for value in self.values[0:self._N]:
            if slopes:
                self.dds.freq_slope(self.dds_idx,value)
            else:
                self.dds.freq(self.dds_idx,value)
            self.dds.exec_at_trg()
            i = i + 1
            if i % self._N_steps_per_dds_write == 0:
                self.dds.write()

        self.dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        self.dds.exec_at_trg()
        self.dds.write()

    @rpc(flags={"async"})
    def write_amp_ramp(self,dt=dv,slopes=False):
        if dt == dv:
            dt = self.p.t_tweezer_amp_ramp_dt

        self.dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        self.dds.trg_timer(dt)
        self.dds.exec_at_trg()
        self.dds.write()

        i = 0

        for value in self.values[0:self._N]:
            if slopes:
                self.dds.amp_slope(self.dds_idx,value)  
            else:
                self.dds.amp(self.dds_idx,value)
            self.dds.exec_at_trg()
            i = i + 1
            if i % self._N_steps_per_dds_write == 0:
                self.dds.write()

        self.dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        self.dds.exec_at_trg()
        self.dds.write()

class TweezerController():

    def __init__(self,
                 awg_ip='TCPIP::192.168.1.83::inst0::INSTR',
                  awg_trg_ttl=TTL,
                  tweezer_xmesh=mesh(),
                  expt_params=ExptParams(),
                  core=Core):
        """Controls the tweezers.

        Args:
            sw_ttl (TTL): TTL
            awg_trg_ttl (TTL): TTL
        """        
        self.awg_trg_ttl = awg_trg_ttl
        self.params = expt_params
        self._awg_ip = awg_ip
        self.core = core

        self.tweezer_xmesh = tweezer_xmesh

        self.params.idx_tweezer = 0
        self.traps = []
        self.traps: list[TweezerTrap]
        self.traps_saved = []
        self.traps_saved: list[TweezerTrap]

    @kernel
    def cubic_move(self,tweezer_idx,
                   t_move,x_move,
                   dt=dv,trigger=True):
        if dt == dv:
            dt = self.params.t_tweezer_movement_dt
        self.traps[tweezer_idx].cubic_move(t_move,x_move,
                                           trigger=trigger)

    @kernel
    def sine_move(self,tweezer_idx,
                  t_mod,x_mod,f_mod,
                  t_xmod_ramp=0.,
                  dt=dv,trigger=True):
        if dt == dv:
            dt = self.params.t_tweezer_movement_dt
        self.traps[tweezer_idx].sine_move(t_mod,x_mod,f_mod,t_xmod_ramp,
                                          trigger=trigger)

    @kernel
    def linear_amplitude_ramp(self,tweezer_idx,
                              t_ramp,amp_f,
                              dt=dv,trigger=True):
        if dt == dv:
            dt = self.params.t_tweezer_amp_ramp_dt
        self.traps[tweezer_idx].linear_amplitude_ramp(t_ramp,amp_f,
                                                      trigger=trigger)

    def save_trap_list(self):
        from copy import deepcopy
        self.traps_saved = deepcopy(self.traps)

    def add_tweezer_list(self,
                         position_list=dv_array,
                         amplitude_list=dv_array,
                         cateye_list=db_array,
                         frequency_list=dv_array):
        """Populates the trap list (awg_tweezer.tweezer.traps) with a
        TweezerTrap object for each position (or frequency) and amplitude pair
        provided.

        Args:
            position_list (np.ndarray or float): A list of positions for the
            tweezers. Can be omitted if specifying AOD frequencies.
            amplitude_list (np.ndarray or float): A list of amplitudes for the
            dds tones for each tweezer, in the same order as the position (or
            frequency) list. The total amplitude for all tweezer traps must be
            less than 1.
            cateye_list (np.ndarray or bool): A list of booleans, describing
            whether or not each tweezer is cateye or non-cateye. Unnecessary if
            tweezers are specified by frequency (instead of position).
            frequency_list (np.ndarray or float): A list of frequencies, can be
            provided instead of positions to specify the tweezers by AOD
            frequency.

        Returns:
            List[TweezerTrap]: the list of tweezer traps added.
        """    
        
        def arrcast(v,dtype=float):
            if not (isinstance(v,np.ndarray) or isinstance(v,list)):
                v = [v]
            return np.array(v,dtype=dtype)
        
        position_list = arrcast(position_list)
        amplitude_list = arrcast(amplitude_list)
        cateye_list = arrcast(cateye_list,bool)
        frequency_list = arrcast(frequency_list)
        
        x_specified = np.all(position_list != dv_array)
        amp_specified = np.all(amplitude_list != dv_array)
        cateye_specified = np.all(cateye_list != [db_array])
        freq_specified = np.all(frequency_list != dv_array)

        mesh = self.tweezer_xmesh

        if not x_specified:
            if not freq_specified:
                frequency_list = arrcast(self.params.frequency_tweezer_list)
            position_list = mesh.f_to_x(frequency_list)
            cateye_list = (frequency_list < mesh.f_ce_max)
        if not amp_specified:
            amplitude_list = arrcast(self.params.amp_tweezer_list)
        if x_specified and not cateye_specified:
            raise ValueError('You must indicate cateye/non-cateye of each tweezer if specifying tweezers by position.')
        if x_specified and freq_specified:
            raise ValueError('You must specify either freuqency or position (not both), or specify neither to use default values.')
        if freq_specified and cateye_specified:
            print("Both frequencies and cateye/non-cateye are specified -- ignoring the cateye list, using frequencies.")
        
        if np.sum(amplitude_list) > 1.:
            raise ValueError(f"The amplitudes in amplitude_list sum to a value >1 ({np.sum(amplitude_list)})")

        tweezer_list = []
        for i in range(len(position_list)):
            x = position_list[i]
            a = amplitude_list[i]
            c = cateye_list[i]
            tweezer = self.add_tweezer(x,a,c)
            tweezer_list.append(tweezer)
        return tweezer_list

    def add_tweezer(self,
                    position=dv,
                    amplitude=dv,
                    cateye:bool=False,
                    frequency=dv) -> TweezerTrap:
        """Creates a TweezerTrap object and adds it to the traps list.

        Args:
            position (float, optional): The position of the tweezer relative to
            the origin (see calibration).
            amplitude (float, optional): The DDS amplitude to be used for this
            tweezer trap. The total of all tweezer dds amplitudes must sum to
            < 1.
            cateye (bool, optional): A boolean indicating whether or not that
            tweezer is formed by the cateye side of the tweezer optics.
            Unnecessary if tweezer specified by frequency.
            frequency (float, optional): Can be optionally specified to specify
            the tweezer by AOD freuqency.
        """        
        ampsum = np.sum([t.amplitude for t in self.traps])
        if ampsum + amplitude > 1.:
            raise ValueError(f"The amplitudes in the trap list sum to a value >1 ({ampsum})")
        
        if (position != dv) and (frequency != dv):
            raise ValueError('You must specify either freuqency or position (not both).')

        tweezer = TweezerTrap(position,
                              amplitude,
                              cateye,
                              frequency,
                              self.awg_trg_ttl,
                              self.params,
                              self.core)
        self.traps.append(tweezer)
        return tweezer

    @kernel
    def reset_traps(self,xvarnames):
        self.core.wait_until_mu(now_mu())
        self.reset_trap_list_rpc(xvarnames)
        self.sync_kernel_trap_list()
        self.set_static_tweezers()
        self.core.break_realtime()

    def reset_trap_list_rpc(self,xvarnames):
        """If the user is scanning the trap amplitudes or frequency list, reset
        the trap list and re-create the TweezerTrap objects with the new
        frequencies/amplitudes. This should be done more intelligently in the
        future.

        Args:
            xvarnames (list[str]): The xvarnames list containing strings of the
            keys for the xvars.
        """        
        if "amp_tweezer_list" in xvarnames or "frequency_tweezer_list" in xvarnames:
            self.traps = []
            self.params.idx_tweezer = 0
            self.add_tweezer_list()
            self.save_trap_list()

    @kernel
    def sync_kernel_trap_list(self):
        for idx in range(len(self.traps)):
            self.traps[idx].update_amp(self.get_trap_amp(idx))
            self.traps[idx].update_x(self.get_trap_position(idx))

    def get_trap_amp(self,idx) -> TFloat:
        return self.traps_saved[idx].amplitude
    
    def get_trap_position(self,idx) -> TFloat:
        return self.traps_saved[idx].position
    
    def awg_init(self,two_d = False):
        """Connects to spectrum AWG, sets full-scale voltage amplitude, initializes trigger mode.
        """        
        self.card = spcm.Card(self._awg_ip)

        self.card.open(self._awg_ip)

        # self.card.reset()

        # setup card for DDS
        self.card.card_mode(spcm.SPC_REP_STD_DDS)

        # Setup the channels
        channels = spcm.Channels(self.card)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(0.428 * units.V)
        # channels.amp(1. * units.V)
        self.card.write_setup()

        # trigger mode
        trigger = spcm.Trigger(self.card)
        trigger.or_mask(spcm.SPC_TMASK_EXT0) # disable default software trigger
        trigger.ext0_mode(spcm.SPC_TM_POS) # positive edge
        trigger.ext0_level0(1.5 * units.V) # Trigger level is 1.5 V (1500 mV)
        trigger.ext0_coupling(spcm.COUPLING_DC) # set DC coupling
        self.card.write_setup()

        # Setup DDS functionality
        self.dds = spcm.DDSCommandList(self.card)
        self.dds.reset()

        for trap in self.traps:
            trap.dds = self.dds

        self.dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)
        self.dds.mode = self.dds.WRITE_MODE.WAIT_IF_FULL

        self.dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)

        # thanks jp
        self.core_list = [hex(2**n) for n in range(20)]

        # assign dds cores to channel
        if two_d:
            self.dds.cores_on_channel(1, spcm.SPCM_DDS_CORE8,spcm.SPCM_DDS_CORE9,spcm.SPCM_DDS_CORE10,spcm.SPCM_DDS_CORE11)

        self.dds.write_to_card()

        # Start command including enable of trigger engine
        self.card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    def set_static_tweezers(self, freq_list=[0.], amp_list=[0.], phase_list=[0.]):
        """Sets a static tweezer array. If no arguments are provided,
        information is drawn from the class attribute "traps", which contains
        TweezerTrap objects.

        Args:
            freq_list (ndarray,optional): array of frequencies in Hz
            amp_list (ndarray,optional): array of amplitudes (min=0, max=1)
            phase_list (ndarray,optional): array of phases.
        """
        if np.all(freq_list == [0.]) and np.all(amp_list == [0.]):
            freq_list = [t.frequency for t in self.traps]
            amp_list = [t.amplitude for t in self.traps]

        if np.all(phase_list == [0.]):
            phase_list = self.compute_tweezer_phases(amp_list)
        
        if len(freq_list) != len(amp_list):
            raise ValueError('Amplitude and frequency lists are not of equal length')

        for tweezer_idx in range(len(self.core_list)):
            if tweezer_idx < len(freq_list):
                self.dds[tweezer_idx].amp(amp_list[tweezer_idx])
                self.dds[tweezer_idx].freq(freq_list[tweezer_idx])
                self.dds[tweezer_idx].phase(phase_list[tweezer_idx])
            else:
                pass
        self.dds.exec_at_trg()
        self.dds.write()

    def set_static_2d_tweezers(self, freq_list1=[0.], amp_list1=[0.], phase_list1=[0.],freq_list2=[0.], amp_list2=[0.], phase_list2=[0.]):
        """Sets a static tweezer array. If no arguments are provided,
        information is drawn from the class attribute "traps", which contains
        TweezerTrap objects.

        Args:
            freq_list (ndarray,optional): array of frequencies in Hz
            amp_list (ndarray,optional): array of amplitudes (min=0, max=1)
            phase_list (ndarray,optional): array of phases.
        """
        phase_list1 = self.compute_tweezer_phases(amp_list1)
        phase_list2 = self.compute_tweezer_phases(amp_list2)

        if len(freq_list1) != len(amp_list1):
            raise ValueError('Amplitude and frequency lists are not of equal length')
        if len(freq_list2) != len(amp_list2):
            raise ValueError('Amplitude and frequency lists are not of equal length')

        for tweezer_idx in range(len(freq_list1)):
            if tweezer_idx < len(freq_list1):
                self.dds[tweezer_idx].amp(amp_list1[tweezer_idx])
                self.dds[tweezer_idx].freq(freq_list1[tweezer_idx])
                self.dds[tweezer_idx].phase(phase_list1[tweezer_idx])
            else:
                pass

        for tweezer_idx in range(len(freq_list2)):
            if tweezer_idx < len(freq_list2):
                tweezer_idx = tweezer_idx + 8
                self.dds[tweezer_idx].amp(amp_list2[tweezer_idx-8])
                self.dds[tweezer_idx].freq(freq_list2[tweezer_idx-8])
                self.dds[tweezer_idx].phase(phase_list2[tweezer_idx-8])
            else:
                pass

    def set_amp(self,tweezer_idx,amp,trigger=True):
        self.dds[tweezer_idx].set_amp(amp,trigger)
    
    def reset_awg(self):
        self.dds.reset()

    def compute_tweezer_phases(self,amplitudes):
        phases = np.zeros([len(amplitudes)])
        total_amp = np.sum(amplitudes)
        for tweezer_idx in range(len(amplitudes)):
            if tweezer_idx == 0:
                phases[0] =  360.
            else:
                phase_ij = 0
                for j in range(1,tweezer_idx):
                    phase_ij = phase_ij + 2*np.pi*(tweezer_idx - j)*(amplitudes[tweezer_idx] / total_amp)
                phase_i = (phase_ij % 2*np.pi) * 360
                phases[tweezer_idx] = phase_i
        return phases
    
    @kernel
    def trigger(self):
        self.awg_trg_ttl.pulse(1.e-6)

    
