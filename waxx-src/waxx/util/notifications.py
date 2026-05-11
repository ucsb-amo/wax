"""
waxx.util.notifications
=======================
Lightweight email notification helpers shared across waxx and kexp.

Credentials are read from a two-line plain-text file on the shared Google Drive
(configured via ``waxx.config.ip.EMAIL_CREDENTIALS_FILEPATH``):

    line 1 – sender Gmail address
    line 2 – Gmail app password

This keeps secrets out of any public repository.
"""

import os
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from waxx.config.ip import EMAIL_CREDENTIALS_FILEPATH

logger = logging.getLogger(__name__)

_SMTP_SERVER = "smtp.gmail.com"
_SMTP_PORT = 587


def _load_credentials(credentials_filepath=None):
    """Read (sender_email, app_password) from a two-line text file."""
    path = credentials_filepath or EMAIL_CREDENTIALS_FILEPATH
    # Use utf-8-sig so files written with a UTF-8 BOM (common on Windows)
    # do not leak BOM bytes into the email address/password.
    with open(path, 'r', encoding='utf-8-sig') as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"Credential file must contain at least two non-empty lines "
            f"(sender email, app password): {path}"
        )
    return lines[0], lines[1]


def send_email(recipient, subject, body, credentials_filepath=None):
    """Send a plain-text email via Gmail SMTP.

    Parameters
    ----------
    recipient : str
        Destination email address.
    subject : str
        Email subject line.
    body : str
        Plain-text message body.
    credentials_filepath : str, optional
        Path to credential file.  Defaults to
        ``waxx.config.ip.EMAIL_CREDENTIALS_FILEPATH``.
    """
    sender_email, sender_password = _load_credentials(credentials_filepath)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(_SMTP_SERVER, _SMTP_PORT)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, recipient, msg.as_string())
    server.quit()


def send_run_done_email(
    run_id,
    experiment_filename,
    timestamp=None,
    recipient='herberthearsall@gmail.com',
    credentials_filepath=None,
):
    """Send a run-completion notification email.

    Subject format: ``run {run_id} done: {experiment basename} - {timestamp}``

    Intended to be called from ``analyze()`` immediately after ``self.end()``.
    Failures are caught and logged as warnings so they never abort analysis.

    Example::

        def analyze(self):
            import os
            self.end(os.path.abspath(__file__))
            from waxx.util.notifications import send_run_done_email
            send_run_done_email(self.run_info.run_id, os.path.abspath(__file__))

    Parameters
    ----------
    run_id : int
        The run ID (``self.run_info.run_id`` inside an experiment).
    experiment_filename : str
        Absolute path to the experiment script (pass ``os.path.abspath(__file__)``).
    timestamp : str, optional
        Timestamp string for the subject.  Defaults to the current local time
        formatted as ``YYYY-MM-DD HH:MM:SS``.
    recipient : str, optional
        Destination address.  Defaults to ``herberthearsall@gmail.com``.
    credentials_filepath : str, optional
        Override the default credential file path.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    basename = os.path.basename(experiment_filename)
    subject = f"run {run_id} done: {basename} - {timestamp}"
    try:
        send_email(recipient, subject, subject, credentials_filepath)
        logger.info("Run-done notification sent to %s: %s", recipient, subject)
    except Exception as exc:
        logger.warning("Failed to send run-done notification: %s", exc)
