"""Minimal Zabbix JSON-RPC client (read-only use).

Stdlib only (``urllib``), so ``waxa`` picks up no new dependency.  The
transport is one method, :meth:`ZabbixAPI._post`, which tests replace with
a canned fake.

Authentication: Zabbix 7 takes the session token in an
``Authorization: Bearer`` header.  The lab server's ``guest`` account can log
in with an empty password and has read access to every host, which is all
this module ever needs.  An expired session (``Session terminated`` /
``Not authorised``) is re-established once and the call retried.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://weldlabaio1.physics.ucsb.edu/api_jsonrpc.php"

_REAUTH_MARKERS = ("Session terminated", "Not authorised", "Not authorized")


class ZabbixError(RuntimeError):
    """The server answered with a JSON-RPC error object."""

    def __init__(self, method: str, error: dict):
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message", "")
        self.data = error.get("data", "")
        super().__init__(f"{method}: {self.message} {self.data}".strip())


class ZabbixAPI:
    """JSON-RPC session against one Zabbix frontend.

    Parameters
    ----------
    url : str
        The ``api_jsonrpc.php`` endpoint.
    username, password : str
        Credentials; the defaults are the guest login.
    timeout : float
        Per-request socket timeout in seconds.
    """

    def __init__(self, url: str = DEFAULT_URL, username: str = "guest",
                 password: str = "", timeout: float = 15.0):
        self.url = url
        self.username = username
        self.password = password
        self.timeout = float(timeout)
        self.token: str | None = None
        self._id = 0

    # -- transport -------------------------------------------------------

    def _post(self, payload: dict, token: str | None) -> dict:
        """One HTTP round-trip.  Returns the decoded JSON-RPC envelope."""
        headers = {"Content-Type": "application/json-rpc"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Zabbix at {self.url} unreachable ({exc.reason}). "
                "Are you on the Broida VPN?"
            ) from exc

    def _envelope(self, method: str, params) -> dict:
        self._id += 1
        return {"jsonrpc": "2.0", "method": method, "params": params, "id": self._id}

    # -- public ----------------------------------------------------------

    def version(self) -> str:
        """Server version; needs no login."""
        return self._raw_call("apiinfo.version", {}, auth=False)

    def login(self) -> str:
        self.token = self._raw_call(
            "user.login", {"username": self.username, "password": self.password},
            auth=False,
        )
        return self.token

    def logout(self) -> None:
        if self.token:
            try:
                self._raw_call("user.logout", [], auth=True)
            finally:
                self.token = None

    def call(self, method: str, params):
        """Authenticated call; logs in lazily and once more on session expiry."""
        if self.token is None:
            self.login()
        try:
            return self._raw_call(method, params, auth=True)
        except ZabbixError as exc:
            if any(m in (exc.data or exc.message) for m in _REAUTH_MARKERS):
                self.login()
                return self._raw_call(method, params, auth=True)
            raise

    def _raw_call(self, method: str, params, *, auth: bool):
        reply = self._post(self._envelope(method, params), self.token if auth else None)
        if "error" in reply:
            raise ZabbixError(method, reply["error"])
        return reply["result"]

    # context manager: logs out on exit
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.logout()
        except Exception:  # noqa: BLE001 - logout is best effort
            pass
        return False


__all__ = ["DEFAULT_URL", "ZabbixAPI", "ZabbixError"]
