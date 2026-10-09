"""The run queue's client (waxx.util.device_state.run_queue_client): one
method per request, refusals and silence, "no queue", and follow().  The
monitor server is a fake transport in front of a real RunQueue (with liveOD,
the job processes and the clock faked as in test_run_queue); every request
goes through a JSON round trip, as on the wire.  Nothing is launched, no
socket is opened and the discovery module is never imported."""
import io
import json
import sys
import types

import pytest

from test_run_queue import make_queue
from waxx.util.device_state.run_queue_client import (
    NO_QUEUE_MSG, NoRunQueue, RunQueueClient, RunQueueError, safe_write)

ACTIONS = {"submit": "submit", "cancel": "cancel", "list": "list", "describe": "describe",
           "tail": "tail", "pause": "pause", "resume": "resume", "hold": "hold_request",
           "release": "release_request"}
#: The fake is stricter than the server (which requires owner on cancel,
#: release and resume): the client must send it on every change.
OWNER_REQUIRED = ("submit", "cancel", "pause", "resume", "hold", "release")


class FakeServer:
    """The monitor server as a client sees it: ``request(obj, timeout,
    attempts)`` -> reply dict or None (silence), ``get_status()``.
    ``silent``: how many coming requests get no answer; ``known``: False for an
    older server without the run queue; ``on_request(obj)``: called first, to
    move the world on (or raise KeyboardInterrupt, as Ctrl-C would)."""

    def __init__(self, q):
        self.q = q
        self.requests = []
        self.silent = 0
        self.known = True
        self.on_request = None
        self.run_pending = None
        self.run_loops = {}

    def request(self, obj, timeout=5.0, attempts=2):
        obj = json.loads(json.dumps(obj))
        self.requests.append((obj, attempts))
        if self.on_request is not None:
            self.on_request(obj)
        if self.silent:
            self.silent -= 1
            return None
        if obj.get("type") != "run_queue" or not self.known:
            return {"status": "error", "msg": f"unknown type {obj.get('type')}"}
        method = ACTIONS.get(obj.get("action"))
        if obj.get("action") in OWNER_REQUIRED and obj.get("owner") not in ("person", "agent"):
            return {"status": "error", "msg": "owner is required (person or agent) "
                                              f"(fake server, {obj.get('action')})"}
        if method is None:
            return {"status": "error", "msg": f"unknown run_queue action {obj.get('action')!r}"}
        return json.loads(json.dumps(getattr(self.q, method)(obj)))   # as the server

    def get_status(self):
        if self.silent:
            self.silent -= 1
            return None
        status = {"state": 0, "state_name": "READY", "run_pending": self.run_pending,
                  "run_loops": self.run_loops}
        if self.known:
            status.update(run_queue=self.q.info(), person_hold=self.q.hold.info())
        return json.loads(json.dumps(status))

    def actions(self):
        return [o.get("action") for o, _ in self.requests]


class Script:
    """Steps run one per ``tail`` request (the world moving on while a
    client follows); ``interrupt_at``: tail request numbers that raise
    KeyboardInterrupt instead."""

    def __init__(self, server, steps=(), interrupt_at=()):
        self.server, self.steps, self.interrupt_at = server, list(steps), set(interrupt_at)
        self.n = 0
        server.on_request = self

    def __call__(self, obj):
        if obj.get("action") != "tail":
            return
        self.n += 1
        if self.n in self.interrupt_at:
            raise KeyboardInterrupt
        if self.steps:
            self.steps.pop(0)()


def run_steps(q, run_id=101, outcome="saved", code=0, lines=("shot 1/2", "shot 2/2")):
    """Steps of one job: launch, print the run id and a shot, end."""
    def launch():
        q.tick()

    def start():
        q.spawner.procs[-1].write(f"Run ID: {run_id}", lines[0])
        q.live.start_run(run_id)
        q.tick()

    def end():
        q.spawner.procs[-1].write(*lines[1:])
        q.live.end_run(run_id, outcome)
        q.spawner.procs[-1].code = code
        q.tick()

    return [launch, start, end]


@pytest.fixture
def expts(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("rabi", "tof", "monitor"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    return folder


@pytest.fixture
def q(tmp_path, expts):
    return make_queue(tmp_path, expts)


@pytest.fixture
def server(q):
    return FakeServer(q)


@pytest.fixture
def client(server):
    return RunQueueClient(server, by="jp@kong", sleep=lambda s: None)


# --- requests --------------------------------------------------------------------------

def test_every_action_sends_its_request(client, server, expts):
    reply = client.submit(str(expts / "rabi.py"), argv=["-a", "x=1"], priority=3,
                          owner="agent", label="r1", write_back=False)
    assert reply["ids"] == [1] and reply["jobs"][0]["owner"] == "agent"
    sent, attempts = server.requests[-1]
    assert attempts == 1                                       # a change is never re-sent
    assert sent == {"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py"),
                    "argv": ["-a", "x=1"], "label": "r1", "owner": "agent", "priority": 3,
                    "after": [], "repeat": 1, "write_back": False, "allow_drift": False,
                    "by": "jp@kong"}
    token = reply["jobs"][0]["token"]
    assert client.list()["next"] == [1]
    assert client.describe(1, token)["job"]["label"] == "r1"
    assert client.tail(1, token, 0)["state"] == "queued"
    assert client.pause("all", "lunch")["run_queue"]["paused"]["all"]["reason"] == "lunch"
    assert client.resume("all")["status"] == "ok"
    assert client.hold("mine")["person_hold"]["by"] == "jp@kong"
    assert client.release()["released"]["reason"] == "mine"
    assert client.cancel(1, token=token, queued_only=True)["job"]["state"] == "cancelled"
    retried = {o["action"]: a for o, a in server.requests}
    assert retried == {"submit": 1, "list": 2, "describe": 2, "tail": 2, "pause": 1,
                       "resume": 1, "hold": 1, "release": 1, "cancel": 1}
    assert server.requests[-1][0]["queued_only"] is True
    assert client.status()["run_queue"]["counts"]["cancelled"] == 1


def test_a_relative_path_is_made_absolute_here(client, server, expts, monkeypatch):
    monkeypatch.chdir(expts)
    client.submit("rabi.py")
    assert server.requests[-1][0]["path"] == str(expts / "rabi.py")


def test_a_refusal_raises_with_the_servers_message(client, expts):
    with pytest.raises(RunQueueError, match="no such file") as info:
        client.submit(str(expts / "nothing.py"))
    assert info.value.reply["status"] == "error"
    client.submit(str(expts / "rabi.py"))
    client.cancel(1)
    with pytest.raises(RunQueueError, match="already cancelled"):
        client.cancel(1)


def test_silence_raises_and_a_change_may_have_happened(client, server, expts):
    server.silent = 1
    with pytest.raises(RunQueueError, match="may have acted") as info:
        client.submit(str(expts / "rabi.py"))
    assert info.value.reply is None
    server.silent = 1
    with pytest.raises(RunQueueError, match="did not answer") as info:
        client.list()
    assert "may have acted" not in str(info.value)


def test_no_monitor_server_beaconing_is_no_run_queue(monkeypatch):
    def nobody(timeout):
        raise RuntimeError("[NetClient] server 'monitor:76' not found within 3.0 s")

    with pytest.raises(NoRunQueue) as info:
        RunQueueClient(connect=nobody)
    assert str(info.value).startswith(NO_QUEUE_MSG) and "monitor:76" in str(info.value)
    # the default connect, with the discovery module replaced (never imported here)
    fake = types.ModuleType("waxx.util.comms_server.comm_client")

    class MonitorClient:
        def __init__(self, discovery_timeout=3.0):
            raise RuntimeError("not found")

    fake.MonitorClient = MonitorClient
    monkeypatch.setitem(sys.modules, "waxx.util.comms_server.comm_client", fake)
    with pytest.raises(NoRunQueue, match="not found"):
        RunQueueClient()


def test_a_ctrl_c_while_writing_never_repeats_a_line_on_resume(client, server, q, expts):
    jid = client.submit(str(expts / "rabi.py"))["ids"][0]
    Script(server, run_steps(q))

    class Out(io.StringIO):
        armed = True

        def write(self, text):
            if self.armed and text.startswith("shot 1/2"):
                self.armed = False
                raise KeyboardInterrupt
            return super().write(text)

    out, cursor = Out(), {}
    with pytest.raises(KeyboardInterrupt):
        client.follow(jid, out=out, cursor=cursor)
    assert client.follow(jid, out=out, cursor=cursor)["state"] == "saved"
    lines = out.getvalue().splitlines()
    assert len(lines) == len(set(lines)) and lines.count("Run ID: 101") == 1
    assert lines[-1] == "shot 2/2"                     # "shot 1/2" was lost, not repeated


def test_an_older_monitor_server_without_a_queue_is_no_run_queue(client, server):
    server.known = False
    with pytest.raises(NoRunQueue, match="no run queue"):
        client.list()
    with pytest.raises(NoRunQueue):
        client.status()


# --- follow -------------------------------------------------------------------------------

def test_follow_copies_the_log_and_returns_the_final_job(client, server, q, expts):
    jid = client.submit(str(expts / "rabi.py"))["ids"][0]
    Script(server, [lambda: None] + run_steps(q))
    out, waits = io.StringIO(), []
    job = client.follow(jid, out=out, on_wait=waits.append, wait_poll_s=0.0)
    assert job["state"] == "saved" and job["run_id"] == 101
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("-- run queue ") and lines[1].startswith("$ ")
    assert lines[2:] == ["Run ID: 101", "shot 1/2", "shot 2/2"]
    assert waits and waits[0]["waiting"] == "launching"
    # follow was given only the id: it took the token once and sent it with every tail
    tails = [o for o, _ in server.requests if o["action"] == "tail"]
    assert tails and all(o.get("token") == job["token"] for o in tails)


def test_follow_goes_on_from_its_cursor_after_an_interrupt(client, server, q, expts):
    jid = client.submit(str(expts / "rabi.py"))["ids"][0]
    script = Script(server, run_steps(q), interrupt_at={3})
    out, cursor = io.StringIO(), {}
    with pytest.raises(KeyboardInterrupt):
        client.follow(jid, out=out, cursor=cursor)
    assert cursor["offset"] > 0 and cursor["state"] == "running"
    script.interrupt_at = set()
    job = client.follow(jid, out=out, cursor=cursor)
    assert job["state"] == "saved"
    lines = out.getvalue().splitlines()
    assert lines.count("Run ID: 101") == 1 and lines[-2:] == ["shot 1/2", "shot 2/2"]


def test_follow_waits_out_silence_and_gives_up_after_lost_s(client, server, q, expts):
    jid = client.submit(str(expts / "rabi.py"))["ids"][0]
    Script(server, run_steps(q))
    server.silent = 3
    notes = []
    job = client.follow(jid, out=io.StringIO(), on_lost=notes.append)
    assert job["state"] == "saved"
    assert len(notes) == 2 and "not answering" in notes[0] and "again" in notes[1]
    t = [0.0]

    def clock():
        t[0] += 100.0
        return t[0]

    server.silent = 10 ** 6
    with pytest.raises(RunQueueError, match="has not answered for 300 s"):
        client.follow(jid, out=io.StringIO(), lost_s=300.0, clock=clock)


def test_follow_stops_on_a_refusal(client, server, expts):
    client.submit(str(expts / "rabi.py"))
    with pytest.raises(RunQueueError, match="has token"):
        client.follow(1, token="nope", out=io.StringIO())


def test_a_line_the_terminal_cannot_show_does_not_end_the_follow(client, server, q, expts):
    jid = client.submit(str(expts / "rabi.py"))["ids"][0]
    Script(server, run_steps(q, lines=("t = 5 µs → ok", "done")))
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    assert client.follow(jid, out=out)["state"] == "saved"
    out.flush()
    text = raw.getvalue().decode("ascii")
    assert "t = 5 ?s ? ok" in text and text.endswith("done\n")


def test_safe_write():
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    safe_write(buf, "5 µs →\n")
    buf.flush()
    assert buf.buffer.getvalue() == "5 µs ?\n".encode("cp1252")
