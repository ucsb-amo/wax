"""A stand-in driver for the connection agent tests (test_connection_agent.py):
no hardware, only prints.  Its prints go to stdout on purpose -- the agent
must keep them off the protocol."""
import time


class FakeDriver:
    COMMANDS = ("echo", "boom")

    def __init__(self, open_block_s=0., close_ok=True, fail_build=False):
        if fail_build:
            raise RuntimeError("no such device")
        self.is_opened = False
        self.open_block_s = float(open_block_s)
        self.close_ok = bool(close_ok)
        print("fake: built")

    def open(self):
        print("fake: opening")
        time.sleep(self.open_block_s)
        self.is_opened = True

    def close(self):
        print("fake: closed")
        self.is_opened = False
        return self.close_ok

    def is_open(self):
        return self.is_opened

    def detail(self):
        return "fake device · ok"

    def echo(self, value):
        return {"echo": value}

    def boom(self):
        raise ValueError("boom")
