"""The connection to a Spectrum AWG card in a netbox: opening it, with retries
while another connection holds the card, and stopping and closing it without
ever hanging the caller.

The netbox takes one connection at a time, and the driver's calls have no
timeouts of their own. AwgConnection owns ``self.card``; a subclass says how
to set the card up once it is open (the ``setup`` passed to connect).
TweezerController (spectrum_DDS_tweezer) is built on it. Host-side only.
"""

import atexit
import re
import socket
import threading
import time
import urllib.request

import spcm

# Driver error texts that mean "the card is there, but the connection could not
# be made right now" -- usually the previous run has not let go of it yet. The
# AWG is a network device and accepts only one connection at a time.
RETRYABLE_AWG_ERROR_TEXTS = ('in use','locked','network','timeout','not found','no connection')
# The subset that means another connection has the card.
AWG_IN_USE_ERROR_TEXTS = ('in use','locked')

# How long connect keeps trying while another connection has the card. A run
# whose core connection has just been taken by a new run lets go of the card
# only once its process notices and exits, which can take longer than the ~4 s
# (3 tries) this used to allow -- and the new run then failed.
T_AWG_IN_USE_WAIT = 30.
T_AWG_RETRY_INTERVAL = 2.
# Tries for the other retryable errors (network, timeout, ...).
N_AWG_RETRIES = 3

# How long close() waits for the card to stop and the connection to close
# before it gives up on them. The driver calls have no timeout of their own:
# spcm_vClose ends by joining a driver worker thread with an infinite wait, and
# on 2026-09-26 (run 83101) that join never returned -- post_scan never
# finished and the run was never saved.
T_AWG_CLOSE_TIMEOUT = 5.
# The same for the exit handler, before the process exits anyway.
T_AWG_EXIT_CLOSE_TIMEOUT = 5.
# A close that returns but takes longer than this is reported, to show how
# near the timeout the closes that do return come.
T_AWG_CLOSE_SLOW = 1.

class AwgConnectionError(Exception):
    """Raised when the connection to the AWG could not be opened."""
    pass

def awg_driver_error_text(handle=None):
    """Returns the driver's description of the last error.

    Passing no handle asks the driver for the last error overall, which is how
    the reason for a failed open is read back (there is no handle to ask).
    """
    try:
        error = spcm.SpcmError()
        error._handle = handle
        error.get_info()
        text = str(error).strip()
    except Exception:
        text = ""
    return text or "no reason reported by the driver"

def awg_error_text(e):
    """Returns a readable message for an exception raised by the spcm driver."""
    text = str(e).strip()
    if text and text != "None":
        return text
    return awg_driver_error_text()

def is_retryable_awg_error(e):
    if isinstance(e,AwgConnectionError):
        return True
    text = awg_error_text(e).lower()
    return any(s in text for s in RETRYABLE_AWG_ERROR_TEXTS)

def is_awg_in_use_error(e):
    text = awg_error_text(e).lower()
    return any(s in text for s in AWG_IN_USE_ERROR_TEXTS)

def awg_holder(awg_ip, timeout=2.):
    """Returns the IP address the AWG netbox reports as holding the card
    ('0.0.0.0' when nobody does), or None if that could not be read.

    The netbox's status web page lists each instrument address with "used by
    <ip>"; the driver's own "in use" error does not say by whom.
    """
    host = awg_ip.split('::')[1] if '::' in awg_ip else awg_ip
    try:
        with urllib.request.urlopen(f"http://{host}/status_page.php", timeout=timeout) as r:
            html = r.read().decode('latin-1')
    except Exception:
        return None
    m = (re.search(re.escape(awg_ip) + r".*?used by\s+([^\s<]+)", html, re.DOTALL)
         or re.search(r"used by\s+([^\s<]+)", html))
    return m.group(1) if m else None

def awg_holder_text(awg_ip):
    """Says who holds the AWG, for the "in use" messages."""
    holder = awg_holder(awg_ip)
    if holder is None:
        return "the netbox status page did not say who holds it"
    if holder == '0.0.0.0':
        return "the netbox now reports it free"
    try:
        own = socket.gethostbyname_ex(socket.gethostname())[2]
    except Exception:
        own = []
    if holder in own:
        return f"held by {holder} -- this PC, so another process here"
    return f"held by {holder}"


class AwgConnection():
    """The connection to one Spectrum AWG card: ``self.card`` (an spcm.Card,
    None while closed) at the address ``self._awg_ip``."""

    #: Names the card in terminal messages and errors.
    _awg_label = "awg"
    #: Added to the "still in use" error: where to look for whatever holds it.
    _in_use_advice = ""

    def __init__(self, awg_ip):
        self._awg_ip = awg_ip
        self.card = None

    def connect(self, setup, t_wait_in_use=T_AWG_IN_USE_WAIT):
        """Opens the card and runs ``setup()`` (self.card is set by then).

        The card takes one connection at a time. While another connection has
        it, keeps trying for t_wait_in_use seconds, saying who holds it (see
        awg_holder); other retryable errors get N_AWG_RETRIES tries. A driver
        error in setup counts like one in the open. Registers the exit handler
        that closes the card if nothing else does.
        """
        # If this process already holds the card (a second init_kernel in the
        # same run, e.g. after a warm-up dry run), let go of it first -- the
        # card takes one connection at a time.
        self.close()

        t_give_up = time.monotonic() + t_wait_in_use
        attempt = 0
        holder_said = None

        while True:
            attempt += 1
            try:
                self.card = self._open_card()
                setup()
                self._register_awg_atexit()
                if holder_said is not None:
                    print(f"{self._awg_label} connected (attempt {attempt}).")
                return

            except (AwgConnectionError, spcm.SpcmException) as e:
                reason = str(e) if isinstance(e, AwgConnectionError) else awg_error_text(e)

                # Drop this attempt's connection before trying again, otherwise
                # the retry adds a second connection to a card that only
                # accepts one.
                self.close()

                in_use = is_awg_in_use_error(e)
                if in_use:
                    holder = awg_holder_text(self._awg_ip)
                    reason = f"{reason} ({holder})"
                    retry = time.monotonic() + T_AWG_RETRY_INTERVAL < t_give_up
                else:
                    retry = is_retryable_awg_error(e) and attempt < N_AWG_RETRIES

                if retry:
                    if not in_use:
                        print(f"{self._awg_label} connection failed ({reason}), retrying in {T_AWG_RETRY_INTERVAL} s")
                    elif holder != holder_said:
                        print(f"{self._awg_label} is in use ({holder}); waiting up to "
                              f"{t_wait_in_use:g} s for it to be released")
                        holder_said = holder
                    time.sleep(T_AWG_RETRY_INTERVAL)
                    continue

                if in_use:
                    reason += f" -- still in use after {t_wait_in_use:g} s.{self._in_use_advice}"

                # ARTIQ carries only the type and message of an exception back
                # from the kernel, and SpcmException never passes its message
                # to Exception.__init__ -- so print the reason here, where it
                # reaches the terminal, and re-raise with it in the message.
                print(f"{self._awg_label} init failed: {reason}")
                raise RuntimeError(f"{self._awg_label} init failed: {reason}") from e

    def _open_card(self):
        """Opens the connection to the AWG and returns the card.

        spcm.Card.open stores whatever spcm_hOpen hands back, including a null
        handle when the connection fails, and marks the card as open anyway. If
        we do not check here, the failure only shows up at the first register
        access (card_mode), which makes it look like a card mode problem.
        """
        card = spcm.Card(self._awg_ip)
        card.open(self._awg_ip)
        if not card.handle():
            # Nothing to stop or close -- otherwise Device.__del__ raises on
            # the null handle while it is being garbage collected.
            card._closed = True
            raise AwgConnectionError(
                f"Could not connect to the {self._awg_label} at {self._awg_ip}: {awg_driver_error_text()}")
        return card

    def close(self, timeout=None, when=""):
        """Stops the card, and its DMA, and closes the connection to it.

        Stopping alone leaves the connection claimed. Each step is guarded
        because close() also runs on the failure path, where the card may be
        half set up.

        The driver steps run in a worker thread, and close() waits for them at
        most `timeout` s (T_AWG_CLOSE_TIMEOUT): the driver has no timeout of its
        own, and spcm_vClose has hung for good (run 83101), which stopped the
        run before end(). On a timeout this process lets go of the card anyway
        -- nothing here touches it again -- and says which step did not return.
        The netbox frees the card when this process's connection drops.

        `when` is added to the messages (e.g. " at exit").
        Returns True if the card closed, False if the time ran out.
        """
        card = getattr(self,'card',None)
        if card is None:
            return True
        if timeout is None:
            timeout = T_AWG_CLOSE_TIMEOUT
        # Let go of the card before the driver calls, so that whatever they do,
        # no later close (connect, the exit handler) tries this card again.
        # (Not card._closed yet: spcm refuses every command on a card marked
        # closed, the stop included -- the worker marks it just before
        # spcm_vClose.)
        self.card = None
        progress = {'step': 'stop'}
        worker = threading.Thread(target=self._stop_and_close_card, args=(card, progress),
                                  name="awg-close", daemon=True)
        t0 = time.monotonic()
        worker.start()
        worker.join(timeout)
        elapsed = time.monotonic() - t0
        if worker.is_alive():
            # Given up on: Device.__del__ must not stop/close it at exit.
            card._closed = True
            if progress['step'] == 'stop':
                state = ("stopping the card did not return, so the card may still be"
                         " running")
            else:
                state = ("the card was stopped, but closing the connection"
                         " (spcm_vClose) did not return")
            print(f"{self._awg_label}: *** {state} within {timeout:g} s{when}. Not waiting any"
                  f" longer; the netbox frees the card when this process's connection"
                  f" drops. ***")
            return False
        if elapsed > T_AWG_CLOSE_SLOW:
            print(f"{self._awg_label}: closing the card took {elapsed:.1f} s{when}.")
        return True

    def _stop_and_close_card(self, card, progress):
        """close()'s driver steps, run in its worker thread. progress['step']
        says how far they got ('stop', 'close', 'done')."""
        try:
            # With M2CMD_DATA_STOPDMA: DDS commands go to the card by DMA
            # (SPCM_DDS_DTM_DMA), and Spectrum's manual asks for STOPDMA before
            # teardown on remote (Ethernet) devices; spcm's own close sends it.
            card.stop(spcm.M2CMD_DATA_STOPDMA)
        except Exception as e:
            print(f"{self._awg_label}: stop with STOPDMA failed while closing"
                  f" ({awg_error_text(e)}); trying a plain stop")
            try:
                card.stop()
            except Exception as e:
                print(f"{self._awg_label}: stop failed while closing ({awg_error_text(e)})")
        progress['step'] = 'close'
        # Marked closed before the handle goes, as spcm's own close does, so
        # Device.__del__ does not stop/close it a second time.
        card._closed = True
        try:
            card.close(card.handle())
        except Exception as e:
            print(f"{self._awg_label}: close failed ({awg_error_text(e)})")
        card._handle = None
        progress['step'] = 'done'

    def _register_awg_atexit(self):
        """Releases the AWG connection when the process ends.

        A run that crashes never reaches post_scan, so without this the card
        stays claimed until the interpreter happens to garbage collect it, and
        the next run cannot connect.
        """
        if getattr(self,'_awg_atexit_registered',False):
            return
        atexit.register(self._close_at_exit)
        self._awg_atexit_registered = True

    def _close_at_exit(self, timeout=None):
        """The exit handler: close() with T_AWG_EXIT_CLOSE_TIMEOUT.

        It runs only when nothing closed the card before (for a run: an abort
        or a crash skipped post_scan). A process stuck here also never runs the
        exit handlers registered before this one -- atexit runs them last in,
        first out -- among them the monitor fence's run_withdrawn. If the time
        runs out the process exits anyway; the netbox frees the card when this
        process's connection drops.
        """
        if timeout is None:
            timeout = T_AWG_EXIT_CLOSE_TIMEOUT
        self.close(timeout=timeout, when=" at exit")
