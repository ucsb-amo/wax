# Service discovery moved to the standalone `beacon` package.
# Re-exported here so `from waxx.util.comms_server import NetServer` still works.
from beacon.discovery import NetServer, NetClient, discover, DISCOVERY_PORT

__all__ = ["NetServer", "NetClient", "discover", "DISCOVERY_PORT"]
