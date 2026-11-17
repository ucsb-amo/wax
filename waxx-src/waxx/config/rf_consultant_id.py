"""RF consultants as ARTIQ control objects.

The assignments live in ``beacon.tpi.rf_consultant_id`` (one ``assign`` line
per serial, shared with the TPI GUI and notebooks).  This frame runs them and
then promotes every assigned record to a :class:`RFConsultant`, so an
experiment reaches each consultant by its key::

    self.rf_consultant.raman_rf.set_frequency(80.e6)

A lab frame (``kexp.config.rf_consultant_id``) subclasses this one and may
tighten a limit after ``super().__init__``::

    self.raman_rf.limit(max_amplitude_dbm=-5.)
"""

from artiq.coredevice.core import Core

from beacon.tpi.rf_consultant_base import RfConsultantId
from beacon.tpi.rf_consultant_id import rf_consultant_frame as rf_consultant_frame_beacon

from waxx.control.misc.rf_consultant import RFConsultant


class rf_consultant_frame(rf_consultant_frame_beacon):
    def __init__(self, core=Core):
        self.core = core
        super().__init__()          # setup(), the beacon assignments, cleanup()

    def cleanup(self):
        super().cleanup()           # stamps each attribute name as its key
        for key, val in list(self.__dict__.items()):
            if isinstance(val, RfConsultantId):
                setattr(self, key, RFConsultant(val, core=self.core))

    def consultants(self) -> list:
        """Every assigned consultant, as :class:`RFConsultant`."""
        return [v for v in self.__dict__.values() if isinstance(v, RFConsultant)]
