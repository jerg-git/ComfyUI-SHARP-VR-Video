"""Local-only optional Windows renderer launch; no shell or user commands."""
import ipaddress
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote

from aiohttp import web

log = logging.getLogger(__name__)
_last_launch = None
_COOLDOWN = 20.0
_LOCAL_HOST = re.compile(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?", re.IGNORECASE)


def _local_origin(request):
    """Reject remote peers, non-loopback Host, cross-origin and simple POSTs."""
    transport = request.transport
    peer = transport.get_extra_info("peername") if transport else None
    try:
        addr = ipaddress.ip_address(peer[0])
        if not addr.is_loopback and not (getattr(addr, "ipv4_mapped", None) and addr.ipv4_mapped.is_loopback):
            return None
    except (TypeError, ValueError, IndexError):
        return None
    host = request.headers.get("Host", "")
    match = _LOCAL_HOST.fullmatch(host)
    if not match or (match[1] and not 1 <= int(match[1]) <= 65535):
        return None
    if request.scheme not in ("http", "https"):
        return None
    origin = f"{request.scheme}://{host.lower()}"
    if request.headers.get("Origin", "").lower() != origin:
        return None
    if request.headers.get("X-Sharp-Renderer-Launch") != "1":
        return None
    if request.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
        return None
    return origin


def _find_browser():
    locations = [
        ("ProgramFiles", "Google/Chrome/Application/chrome.exe", "Chrome"),
        ("ProgramFiles(x86)", "Google/Chrome/Application/chrome.exe", "Chrome"),
        ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe", "Chrome"),
        ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe", "Edge"),
        ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe", "Edge"),
    ]
    for env, suffix, kind in locations:
        base = os.environ.get(env)
        if base:
            path = Path(base) / suffix
            if path.is_absolute() and path.is_file():
                return path, kind
    return None, None


def _launch_browser(origin):
    browser, kind = _find_browser()
    if browser is None:
        raise FileNotFoundError("Chrome or Edge was not found in its usual Windows installation folders.")
    local = os.environ.get("LOCALAPPDATA")
    if not local or not Path(local).is_absolute():
        raise RuntimeError("Windows LOCALAPPDATA is unavailable.")
    profile = Path(local) / "SHARP-VR-Video" / "RendererProfile" / kind
    folder = quote(Path(__file__).resolve().parents[1].name, safe="")
    url = f"{origin}/extensions/{folder}/viewer_gaussian.html?batch=1&worker=1&v=sharp-release-030"
    args = [str(browser), f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", "--disable-features=CalculateNativeWinOcclusion",
            "--new-window", url]
    subprocess.Popen(args, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)
    return kind


async def launch_renderer(request, dedicated_active):
    global _last_launch
    def reply(status, message, code=200):
        return web.json_response({"status": status, "message": message}, status=code,
                                 headers={"Cache-Control": "no-store"})
    origin = _local_origin(request)
    if origin is None:
        return reply("error", "Launch is only available from the local ComfyUI browser (localhost or loopback).", 403)
    if os.name != "nt":
        return reply("error", "The optimized launcher is Windows-only. Use Open dedicated SHARP renderer instead.", 400)
    if dedicated_active():
        return reply("active", "A dedicated renderer is already active. To switch to the optimized browser, close it while idle and try again after a few seconds.")
    now = time.monotonic()
    if _last_launch is not None and now - _last_launch < _COOLDOWN:
        return reply("pending", "A renderer was just launched. Check the new browser window and allow it to connect.")
    try:
        kind = _launch_browser(origin)
    except FileNotFoundError as exc:
        return reply("error", str(exc), 404)
    except (OSError, RuntimeError):
        log.exception("[SHARP VR Video] Could not launch dedicated renderer")
        return reply("error", "Could not launch the browser. Try launch_renderer_windows.bat manually.", 500)
    _last_launch = now
    log.info("[SHARP VR Video] Dedicated renderer launch requested (%s)", kind)
    return reply("launched", f"Dedicated {kind} renderer launched. Keep the window open and visible.")
