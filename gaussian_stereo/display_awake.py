"""Scoped Windows display-idle inhibition for active SHARP batch rendering."""
import ctypes
import logging
import os
import sys
from contextlib import contextmanager

_LOG = logging.getLogger(__name__)
_ES_CONTINUOUS = 0x80000000
_ES_DISPLAY_REQUIRED = 0x00000002


def _execution_state_api():
    api = ctypes.WinDLL('kernel32', use_last_error=True).SetThreadExecutionState
    api.argtypes = [ctypes.c_uint32]
    api.restype = ctypes.c_uint32
    return api


@contextmanager
def display_awake():
    """Prevent idle display power-off only while this thread renders a batch.

    Restore the previous thread state, including on exceptions/interrupts.
    No persistent OS setting is changed. Non-Windows hosts are unaffected.
    """
    if sys.platform != 'win32' or os.environ.get('SHARP_VR_KEEP_DISPLAY_AWAKE', '1') == '0':
        yield
        return
    api = None
    previous = 0
    try:
        api = _execution_state_api()
        previous = api(_ES_CONTINUOUS | _ES_DISPLAY_REQUIRED)
        if previous:
            _LOG.info('[SHARP VR Video] Display-awake protection active for this render batch')
        else:
            _LOG.warning('[SHARP VR Video] Windows display-awake request failed; continuing normally')
    except Exception:
        _LOG.warning('[SHARP VR Video] Windows display-awake request unavailable; continuing normally', exc_info=True)
    try:
        yield
    finally:
        if previous and api is not None:
            try:
                if api(_ES_CONTINUOUS | previous):
                    _LOG.info('[SHARP VR Video] Display-awake protection released')
                else:
                    _LOG.warning('[SHARP VR Video] Could not restore thread display state; restart ComfyUI if needed')
            except Exception:
                _LOG.warning('[SHARP VR Video] Could not restore thread display state; restart ComfyUI if needed', exc_info=True)
