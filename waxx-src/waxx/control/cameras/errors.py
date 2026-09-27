class FrameLostError(TimeoutError):
    """A camera reported a frame of the current grab as lost.

    A TimeoutError on purpose: liveOD's camera thread turns a TimeoutError into
    a run that is kept but marked incomplete.  Every frame that did arrive was
    queued under its own hardware index before this is raised, so nothing after
    the lost frame moved into its slot.  ``lost`` holds the lost indices."""
    def __init__(self, message: str, lost=()):
        super().__init__(message)
        self.lost = tuple(int(i) for i in lost)
