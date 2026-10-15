import ipdb
import signal
import traceback as tb
import sys

def ipdb_safety_net():
    """Attaches a "safety net" for unexpected errors in a Python script.

    When called, PDB will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.
    """

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        ipdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        ipdb.post_mortem(traceback)

    sys.excepthook = excepthook
