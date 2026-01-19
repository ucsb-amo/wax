import time
class RunInfo():
    def __init__(self):
        self.run_id = 0
        self.run_datetime = time.localtime(time.time())
        self._run_description = ""
        date = self.run_datetime
        self.run_date_str = time.strftime("%Y-%m-%d", date)
        self.run_datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S", date)
        self.filepath = ""
        self.experiment_filepath = ""
        self.xvarnames = []
        self.imaging_type = 0
        self.save_data = 1
        self.expt_class = "expt"