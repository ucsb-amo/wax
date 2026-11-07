"""Two-loop intensity PID on a qNimble Quarto (firmware e.g.
k-exp/kexp/control/quarto/pid_intensity_stabilize_raman): serial protocol,
server (owns the port, serves it over TCP, logs), client, CLI and dashboard
panel.  Machine-agnostic: the port, server id, loop names and folders come
from the host's launcher."""
