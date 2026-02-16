"""Helper functions for Simulated Annealing ABC (SABC)."""

import logging
import sys
from typing import Iterable

LOG = logging.getLogger(__name__)

try:
    from IPython import get_ipython

    JUPYTER_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except ImportError:
    JUPYTER_NOTEBOOK = False

INTERACTIVE_SESSION = JUPYTER_NOTEBOOK or sys.stderr.isatty() or sys.stdout.isatty()

try:
    from tqdm.auto import tqdm

    _has_tqdm = True
except ImportError:
    _has_tqdm = False
try:
    from rich.progress import track

    _has_rich = True
except ImportError:
    _has_rich = False


def track_progress(iterable, show_progressbar: bool = True) -> Iterable:
    """Track progress of an iterable using tqdm or rich, if available."""
    if not INTERACTIVE_SESSION:
        if show_progressbar:
            LOG.debug(
                "Cannot show progress bar in an interactive session. Progress bars are disabled."
            )
        else:
            LOG.debug("Progress bars are disabled in non-interactive sessions.")
        return iterable

    if show_progressbar and _has_rich:
        return track(
            iterable,
            description="Running population updates",
            transient=True,
        )
    elif show_progressbar and _has_tqdm:
        return tqdm(
            iterable,
            desc="Running population updates",
            leave=False,
        )
    else:
        LOG.debug("Running population updates without progress bars.")
        return iterable
