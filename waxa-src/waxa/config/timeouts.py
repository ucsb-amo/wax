DEFAULT_TIMEOUT = 45.

# scribe timeouts
REMOVE_DATA_POLL_INTERVAL = 0.25
# Bound on remove_incomplete_data()'s "is the file free?" open.  This runs on
# the abort path, so it must stay short: a file that never got its 'data' group
# is exactly what we are tearing down and must not stall teardown.
REMOVE_DATA_TIMEOUT = 10.
CHECK_FOR_DATA_AVAILABLE_PERIOD = 0.05 # 
CHECK_CAMERA_READY_ACK_PERIOD = 0.1 # waiting time if data not avaiable
T_NOTIFY = 5 # prints a message every T_NOTIFY seconds if data not available
N_NOTIFY = T_NOTIFY // CHECK_FOR_DATA_AVAILABLE_PERIOD
