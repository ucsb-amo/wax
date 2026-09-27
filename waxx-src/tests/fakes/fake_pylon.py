"""A fake of the vendor ``pylon`` module, enough for waxx's BaslerUSB.

Installed into sys.modules (as ``<vendor>.pylon``) before loading basler_usb
under a private module name, so the real driver code runs against it.

It mirrors the one behaviour that shapes BaslerUSB: InstantCamera routes any
attribute assignment that is not a dunder name into the GenICam node map, and
an unknown node is an error -- so a driver that tries to keep plain instance
attributes fails here as it would on the real camera.

State per camera lives in ``cam._fake``: ``nodes``, ``writes`` (node, value)
in order, ``calls`` (method names in order), ``results`` (queued grab
results: ("ok", array, timestamp) | ("fail", description) | None = poll
timeout).
"""
from __future__ import annotations

import time

import numpy as np

GrabStrategy_OneByOne = "GrabStrategy_OneByOne"
GrabStrategy_LatestImages = "GrabStrategy_LatestImages"
TimeoutHandling_Return = "TimeoutHandling_Return"


class RuntimeException(Exception):
    pass


class LogicalErrorException(Exception):
    pass


class OutOfRangeException(Exception):
    pass


class _Node:
    def __init__(self, cam, name, value, lo=None, hi=None):
        self._cam, self._name, self._value, self._lo, self._hi = cam, name, value, lo, hi

    def SetValue(self, value):
        if self._lo is not None and not (self._lo <= value <= self._hi):
            raise OutOfRangeException(f"{self._name}: {value} outside [{self._lo}, {self._hi}]")
        self._cam._fake["writes"].append((self._name, value))
        self._value = value

    def GetValue(self):
        return self._value

    def GetMin(self):
        return self._lo

    def GetMax(self):
        return self._hi

    @property
    def Value(self):
        return self._value

    @Value.setter
    def Value(self, value):
        self.SetValue(value)


class _Command:
    def __init__(self, cam, name):
        self._cam, self._name = cam, name

    def Execute(self):
        self._cam._fake["writes"].append((self._name, "Execute"))


class DeviceInfo:
    def __init__(self, serial=""):
        self._serial = serial

    def SetSerialNumber(self, serial):
        self._serial = str(serial)

    def GetSerialNumber(self):
        return self._serial


class _Device:
    def __init__(self, serial, exposure_range=(19.0, 1.0e7), gain_range=(0.0, 24.0)):
        self.serial = serial
        self.exposure_range = exposure_range
        self.gain_range = gain_range


class TlFactory:
    devices: dict = {}
    _instance = None

    @classmethod
    def GetInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def CreateFirstDevice(self, di=None):
        if di is None:
            if not self.devices:
                raise RuntimeException("No device is available.")
            return next(iter(self.devices.values()))
        dev = self.devices.get(di.GetSerialNumber())
        if dev is None:
            raise RuntimeException("No device is available or no device contains the "
                                   "provided device info properties.")
        return dev


def reset(serials=("40316451",), **device_kwargs):
    """Fresh device list for a test."""
    TlFactory.devices = {s: _Device(s, **device_kwargs) for s in serials}
    TlFactory._instance = None


class _GrabResult:
    def __init__(self, kind, payload=None, ts=0):
        self.kind, self.payload, self.TimeStamp = kind, payload, ts
        self.released = False

    def IsValid(self):
        return self.kind != "invalid"

    def GrabSucceeded(self):
        return self.kind == "ok"

    def GetArray(self):
        return self.payload

    def GetErrorDescription(self):
        return self.payload if self.kind == "fail" else ""

    def Release(self):
        self.released = True


class InstantCamera:
    def __init__(self):
        object.__setattr__(self, "_fake", {
            "device": None, "open": False, "grabbing": False, "max": 0, "retrieved": 0,
            "strategy": None, "nodes": {}, "writes": [], "calls": [], "results": [],
        })

    # -- node-map attribute routing (as the vendor InstantCamera does) --------
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        nodes = object.__getattribute__(self, "_fake")["nodes"]
        if name in nodes:
            return nodes[name]
        raise LogicalErrorException(f"Node not existing: {name}")

    def __setattr__(self, name, value):
        if name.startswith("__") or name in ("thisown", "this"):
            object.__setattr__(self, name, value)
            return
        nodes = self._fake["nodes"]
        if name not in nodes:
            raise LogicalErrorException(f"Node not existing: {name} (assignment is routed "
                                        f"to the node map)")
        nodes[name].SetValue(value)

    # -- device -----------------------------------------------------------------
    def _call(self, name):
        self._fake["calls"].append(name)

    def Attach(self, device):
        self._call("Attach")
        f = self._fake
        f["device"] = device
        lo, hi = device.exposure_range
        glo, ghi = device.gain_range
        f["nodes"] = {
            "ExposureTime": _Node(self, "ExposureTime", 1000.0, lo, hi),
            "Gain": _Node(self, "Gain", 0.0, glo, ghi),
            "LineSelector": _Node(self, "LineSelector", "Line1"),
            "LineMode": _Node(self, "LineMode", "Input"),
            "TriggerSelector": _Node(self, "TriggerSelector", "FrameStart"),
            "TriggerMode": _Node(self, "TriggerMode", "Off"),
            "TriggerSource": _Node(self, "TriggerSource", "Line1"),
            "UserSetSelector": _Node(self, "UserSetSelector", "Default"),
            "UserSetLoad": _Command(self, "UserSetLoad"),
        }

    def DestroyDevice(self):
        self._call("DestroyDevice")
        self._fake["open"] = False
        self._fake["device"] = None

    def GetDeviceInfo(self):
        dev = self._fake["device"]
        if dev is None:
            raise RuntimeException("no device attached")
        return DeviceInfo(dev.serial)

    def Open(self):
        self._call("Open")
        if self._fake["device"] is None:
            raise RuntimeException("no device attached")
        self._fake["open"] = True

    def Close(self):
        self._call("Close")
        self._fake["open"] = False

    def IsOpen(self):
        return self._fake["open"]

    # -- grabbing ---------------------------------------------------------------
    def StartGrabbingMax(self, n, strategy):
        self._call("StartGrabbingMax")
        f = self._fake
        f.update(grabbing=True, max=int(n), retrieved=0, strategy=strategy)

    def IsGrabbing(self):
        self._call("IsGrabbing")
        f = self._fake
        return f["grabbing"] and f["retrieved"] < f["max"]

    def RetrieveResult(self, timeout_ms, handling):
        self._call("RetrieveResult")
        f = self._fake
        if not f["results"]:
            time.sleep(min(timeout_ms, 5) * 1e-3)
            return _GrabResult("invalid")
        item = f["results"].pop(0)
        if item is None:
            return _GrabResult("invalid")
        f["retrieved"] += 1
        if item[0] == "ok":
            return _GrabResult("ok", np.asarray(item[1]), item[2] if len(item) > 2 else 0)
        return _GrabResult("fail", item[1])

    def StopGrabbing(self):
        self._call("StopGrabbing")
        self._fake["grabbing"] = False
