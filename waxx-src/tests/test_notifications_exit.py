"""The run-done email must get a chance to finish at process exit, and must
never be able to hang it (its credentials live on a shared drive)."""
import subprocess
import sys
import time

CASE = """
import sys, time, waxx.util.notifications as n
n._EXIT_WAIT_S = 1.0
def fake(recipient, subject, body, cred=None):
    time.sleep({sleep})
    print("MAIL SENT", flush=True)
n.send_email = fake
n.send_run_done_email_async(1, "x.py")
"""


def run(sleep):
    t0 = time.monotonic()
    out = subprocess.run([sys.executable, "-c", CASE.format(sleep=sleep)],
                         capture_output=True, text=True, timeout=30).stdout
    return out, time.monotonic() - t0


def test_a_normal_send_finishes_before_exit():
    out, _ = run(0.3)
    assert "MAIL SENT" in out


def test_a_hung_send_cannot_hold_the_exit():
    out, elapsed = run(3600)
    assert "MAIL SENT" not in out
    assert elapsed < 10
