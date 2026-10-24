"""TweezerController.save_trap_list must not copy the objects the traps share.

Scanning amp_tweezer_list (or frequency_tweezer_list) rebuilds the trap list
every shot (reset_trap_list_rpc) and saves it again. During a run the core's
comm holds an open socket, so a deepcopy of the traps (which reach the core)
raised "cannot pickle 'socket' object" (run 88196). Here the core is a fake
that holds one end of a socketpair; nothing touches the AWG, the network or
an ARTIQ core (AwgConnection.__init__ only stores the address).
"""

import copy
import socket

import numpy as np
import pytest

from waxx.config.expt_params import ExptParams
from waxx.control.tweezer import tweezer_xmesh
from waxx.control.tweezer.spectrum_DDS_tweezer import TweezerController

ADDRESS = "TCPIP::192.0.2.1::inst0::INSTR"   # documentation range, never reached


class FakeCore:
    """Unpicklable like a connected core: its comm holds an open socket."""

    def __init__(self, sock):
        self.comm = sock


class FakeTTL:
    pass


@pytest.fixture
def sockets():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


@pytest.fixture
def controller(sockets):
    params = ExptParams()
    params.frequency_tweezer_list = [72.e6, 79.e6]
    params.amp_tweezer_list = [0.3, 0.4]
    tw = TweezerController(awg_ip=ADDRESS,
                           awg_trg_ttl=FakeTTL(),
                           tweezer_xmesh=tweezer_xmesh(),
                           expt_params=params,
                           core=FakeCore(sockets[0]))
    assert tw.card is None          # nothing was opened
    tw.add_tweezer_list()
    return tw


def test_plain_deepcopy_reproduces_the_bug(controller):
    # the old save_trap_list body, with a core that holds a socket
    with pytest.raises(TypeError, match="pickle"):
        copy.deepcopy(controller.traps)


def test_save_trap_list_snapshots_and_shares_refs(controller):
    controller.save_trap_list()
    traps, saved = controller.traps, controller.traps_saved
    assert len(saved) == len(traps) == 2
    for i, (t, s) in enumerate(zip(traps, saved)):
        assert s is not t
        assert controller.get_trap_amp(i) == t.amplitude
        assert controller.get_trap_position(i) == t.position
        assert s.frequency == t.frequency
        assert s.core is controller.core
        assert s.p is controller.params
        assert s.awg_trig_ttl is controller.awg_trg_ttl
        assert s.mesh is controller.tweezer_xmesh
    assert [controller.get_trap_amp(i) for i in range(2)] == [0.3, 0.4]


def test_saved_values_do_not_follow_the_live_traps(controller):
    # sync_kernel_trap_list restores each shot's start from traps_saved, so a
    # ramp or move that changes a live trap must not change the saved one
    controller.save_trap_list()
    a0 = controller.get_trap_amp(0)
    x0 = controller.get_trap_position(0)
    controller.traps[0].update_amp_rpc(0.05)
    controller.traps[0].update_x_rpc(x0 + 1.e-6)
    assert controller.get_trap_amp(0) == a0
    assert controller.get_trap_position(0) == x0


def test_reset_trap_list_rpc_rebuilds_from_scanned_amp(controller):
    controller.save_trap_list()
    controller.params.amp_tweezer_list = [0.7]
    controller.params.frequency_tweezer_list = [72.e6]
    controller.reset_trap_list_rpc(['amp_tweezer_list'])
    assert len(controller.traps) == 1
    assert controller.get_trap_amp(0) == 0.7
    assert controller.traps_saved[0].core is controller.core
    assert controller.traps[0].dds_idx == 0


def test_reset_trap_list_rpc_leaves_list_when_not_scanned(controller):
    controller.save_trap_list()
    before = controller.traps_saved
    controller.params.amp_tweezer_list = [0.7]
    controller.reset_trap_list_rpc(['t_tof'])
    assert controller.traps_saved is before
    assert controller.get_trap_amp(0) == 0.3
