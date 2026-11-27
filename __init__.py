"""ComfyUI SHARP Gaussian Stereo custom nodes."""

from .gaussian_stereo.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gaussian_stereo.batch_bridge import register_routes
from .gaussian_stereo.queue_guard import try_install as _try_install_queue_guard
try:
    register_routes()
except Exception as exc:
    print(f"[SHARP VR Video] Could not register batch bridge routes: {exc}")

# Best-effort install for QuietNoise Queue Manager. This is intentionally
# silent when Queue Manager is not present. Retry once on server startup in
# case load order prevented installation during import time.
try:
    from server import PromptServer
    if not _try_install_queue_guard():
        async def _startup(_app):
            _try_install_queue_guard()
        PromptServer.instance.app.on_startup.append(_startup)
except Exception:
    pass

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
