"""Everything liveOD does to a run's data file, and nothing else.

  * run_file.py     -- the server's side: reserve the file at INIT_RUN, save into
                       it at END_RUN, delete it when the run is reset or aborted.
  * image_writer.py -- the camera's side: images into the file as they arrive,
                       and deleting the file when the grab dies.

The server, the camera threads and the windows call into this package and keep
the protocol, the run state and the GUI. An edit that can lose or corrupt a run
is an edit here; the lab's guard hook prompts for this path and leaves the rest
of liveOD free.

No imports here: run_file must stay importable without Qt.
"""
