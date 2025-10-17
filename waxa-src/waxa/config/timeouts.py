DEFAULT_TIMEOUT = 20.

# scribe timeouts
REMOVE_DATA_POLL_INTERVAL = 0.25
CHECK_FOR_DATA_AVAILABLE_PERIOD = 0.05 # 
CHECK_CAMERA_READY_ACK_PERIOD = 0.1 # waiting time if data not avaiable
T_NOTIFY = 5 # prints a message every T_NOTIFY seconds if data not available
N_NOTIFY = T_NOTIFY // CHECK_FOR_DATA_AVAILABLE_PERIOD
