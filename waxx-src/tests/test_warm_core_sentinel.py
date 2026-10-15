"""Sentinel for waxx.util.artiq.core: the upstream ARTIQ internals it relies on.

waxx's Core copies upstream ``Core.run`` with one line added (the wait for the run
queue's GO) and gates the connection in ``CommKernel.open``. If ARTIQ changes any of
the functions pinned here, the copy may no longer match what upstream does, or a new
path may connect to the core device without going through ``open()`` -- which on the
lab's core device ends the running experiment. A failure here means: read the upstream
diff, re-audit waxx/util/artiq/core.py (its module docstring lists the dependencies),
then update the hashes.

Offline: no connection is ever made (TEST-NET host, ``create_connection`` patched to
raise).
"""
import ast
import hashlib
import inspect
import textwrap

import pytest

from artiq.coredevice import comm_kernel
from artiq.coredevice.comm_kernel import CommKernel
from artiq.coredevice.core import Core as ArtiqCore

from waxx.util.artiq import core as waxx_core

TEST_HOST = "192.0.2.1"          # TEST-NET-1: never routed

RE_AUDIT = "upstream changed -- re-audit waxx.util.artiq.core"

PINNED = {
    (ArtiqCore, "run"): "7734e645fd02c3de5dcf436da8efb9bf0a0a4f69011a9d52eaaaf94acccf8bbd",
    (ArtiqCore, "_run_compiled"): "046617ff52fe440747633e4265e4a19002cee5b8c69b4cf2000b1d3b6c1253da",
    (ArtiqCore, "compile_and_upload_subkernels"):
        "a4efd246da9fa321501d4a669511ba1787d37ab201ac0bee051d434464d4fdba",
    (CommKernel, "open"): "87fd0c6a554f5e438dec54e58bf8b329e9cfacb344ab0b9c754d861d130193c9",
    (CommKernel, "_read_header"): "d2bacbae08f2c412962932be8c9cc024b2e25180545072443ec7e99ce4a8bf51",
    (CommKernel, "_write_header"): "d525aa610413b74e91b1ccc68c6495bb0614a7fc13dbf6ceb5d34c5c308a4c60",
    # waxx's Core also relies on these two: the constructor builds self.comm =
    # CommKernel(host) without connecting, and CommKernel keeps host / port.
    (ArtiqCore, "__init__"): "c13575f0c331933d0d7ff3a90be63dd4e650496f09c60ca3f148107fbedc63db",
    (CommKernel, "__init__"): "01f7cc58dbd1808d7d015ec7064fc4a7146fdabcde9de12d7a90c65bfbd376a0",
}


def _source(fn):
    return textwrap.dedent(inspect.getsource(fn))


@pytest.fixture
def no_connection(monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError(f"a test tried to connect to {args}")

    monkeypatch.setattr(comm_kernel, "create_connection", refuse)
    return attempts


@pytest.mark.parametrize("owner,name", list(PINNED), ids=[f"{o.__name__}.{n}" for o, n in PINNED])
def test_upstream_source_is_the_audited_one(owner, name):
    digest = hashlib.sha256(inspect.getsource(getattr(owner, name)).encode()).hexdigest()
    assert digest == PINNED[(owner, name)], f"{owner.__name__}.{name}: {RE_AUDIT}"


def _self_calls(fn):
    """Names of the self.<name>(...) calls in fn's body, in source order."""
    tree = ast.parse(_source(fn))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"]
    return [c.func.attr for c in sorted(calls, key=lambda c: (c.lineno, c.col_offset))]


def test_upstream_run_is_compile_then_subkernels_then_run_compiled():
    assert _self_calls(ArtiqCore.run) == [
        "compile", "compile_and_upload_subkernels", "_run_compiled"], RE_AUDIT


def test_armed_run_is_upstream_run_plus_the_wait():
    """The gated branch of waxx's Core.run is upstream's body statement for statement,
    with self._wait_for_go() between the compile and the subkernel upload."""
    upstream = ast.parse(_source(ArtiqCore.run)).body[0].body
    ours = ast.parse(_source(waxx_core.Core.run)).body[0].body
    # skip the two early returns (unarmed, GO already passed)
    assert all(isinstance(s, ast.If) for s in ours[:2])
    gated = ours[2:]
    wait = [s for s in gated if isinstance(s, ast.Expr) and ast.unparse(s) == "self._wait_for_go()"]
    assert len(wait) == 1
    rest = [s for s in gated if s is not wait[0]]
    assert [ast.dump(s) for s in rest] == [ast.dump(s) for s in upstream], RE_AUDIT
    assert _self_calls(waxx_core.Core.run)[-4:] == [
        "compile", "_wait_for_go", "compile_and_upload_subkernels", "_run_compiled"]


def test_compile_has_attribute_writeback_on_by_default():
    sig = inspect.signature(ArtiqCore.compile)
    assert sig.parameters["attribute_writeback"].default is True, RE_AUDIT
    assert "compile" not in vars(waxx_core.Core)    # the profiler wraps the base method


def test_open_is_the_only_place_that_connects():
    src = inspect.getsource(comm_kernel)
    users = [node.name for node in ast.walk(ast.parse(src))
             if isinstance(node, ast.FunctionDef)
             and "create_connection(" in ast.unparse(node)]
    assert users == ["open"], RE_AUDIT
    for name in ("_read_header", "_write_header"):
        assert "self.open()" in _source(getattr(CommKernel, name)), RE_AUDIT


def test_construction_does_not_connect(no_connection):
    comm = CommKernel(TEST_HOST)
    assert (comm.host, comm.port) == (TEST_HOST, 1381)
    assert not hasattr(comm, "socket")
    core = ArtiqCore(None, TEST_HOST, 1e-9)
    assert type(core.comm) is CommKernel and core.comm.core is core
    assert not hasattr(core.comm, "socket")
    assert no_connection == []
