DEFAULT_TIMEOUT = 120.

# camera mother timeouts and intervals
CAMERA_MOTHER_CHECK_DELAY = 0.2
CAMERA_MOTHER_LOG_UPDATE_INTERVAL = 2.
UPDATE_EVERY = CAMERA_MOTHER_LOG_UPDATE_INTERVAL // CAMERA_MOTHER_CHECK_DELAY

# camera, data saver timeouts
INIT_KERNEL_CAMERA_CONNECTION_TIMEOUT = 90.

DATA_SAVER_TIMEOUT = 120.

# Wall-clock cap on how long persistent_get_camera() retries to open a camera
# before giving up and returning a DummyCamera (which triggers a fast
# camera-not-ready handshake failure instead of blocking the run forever).
CAMERA_OPEN_TIMEOUT = 30.

CAMERA_GRAB_TIMEOUT_BASLER_INIT = 20.
CAMERA_GRAB_TIMEOUT_BASLER_RUN = 8.

CAMERA_GRAB_TIMEOUT_ANDOR = 60.

# scribe timeouts
REMOVE_DATA_POLL_INTERVAL = 0.25
CHECK_FOR_DATA_AVAILABLE_PERIOD = 0.05 # 
CHECK_CAMERA_READY_ACK_PERIOD = 0.1 # waiting time if data not avaiable
T_NOTIFY = 5 # prints a message every T_NOTIFY seconds if data not available
N_NOTIFY = T_NOTIFY // CHECK_FOR_DATA_AVAILABLE_PERIOD
