"""Browser-render bridge for the original gsplat viewer."""
import asyncio
import base64
import json
import math
import os
import threading
import time
import uuid
from aiohttp import web
from server import PromptServer
from .renderer_launcher import launch_renderer

_pending = {}
_lock = threading.Lock()
_registered = False
_dedicated_seen = None
_packed_seen = {}
_waiters = set()


def register_routes():
    global _registered
    if _registered:
        return
    routes = PromptServer.instance.routes
    routes.get("/sharp_vr_video_batch/next")(_next)
    routes.post("/sharp_vr_video_batch/complete")(_complete)
    routes.post("/sharp_vr_video_batch/rgba/{job_id}")(_complete_rgba)
    routes.get("/sharp_vr_video_batch/ply/{job_id}.ply")(_ply)
    routes.post("/sharp_vr_video_batch/launch_renderer")(_launch_renderer)
    _registered = True


def _dedicated_active():
    with _lock:
        return _dedicated_seen is not None and time.monotonic() - _dedicated_seen < 10


async def _launch_renderer(request):
    return await launch_renderer(request, _dedicated_active)


async def _next(request):
    global _dedicated_seen
    node_id = request.query.get("node_id")
    shared_worker = request.query.get("worker") == "1"
    dedicated = shared_worker and request.query.get("dedicated") == "1"
    # Long polling keeps browser timers out of the normal frame dispatch path.
    # Short requests from old viewers remain compatible.
    packed_capable = shared_worker and request.query.get("packed") == "1"
    deadline = time.monotonic() + (20.0 if shared_worker and request.query.get("wait") == "1" else 0.0)
    while True:
        with _lock:
            now = time.monotonic()
            if dedicated:
                _dedicated_seen = now
            if packed_capable:
                _packed_seen[dedicated] = now
            prefer_dedicated = (not dedicated and shared_worker and _dedicated_seen is not None
                                and now - _dedicated_seen < 10)
            if not prefer_dedicated:
                for item in _pending.values():
                    if (shared_worker or item["job"]["node_id"] == node_id) and not item["claimed"] and not item["event"].is_set() and (not item["job"].get("packed") or packed_capable):
                        item["claimed"] = True
                        return web.json_response({"job": item["job"]}, headers={"Cache-Control": "no-store"})
        if now >= deadline:
            return web.json_response({"job": None}, headers={"Cache-Control": "no-store"})
        transport = request.transport
        if transport is None or transport.is_closing():
            return web.json_response({"job": None})
        # Event-driven wakeup, retaining a short watchdog to cover the race
        # between checking the queue and registering a waiter.
        event = asyncio.Event()
        waiter = (asyncio.get_running_loop(), event)
        with _lock:
            _waiters.add(waiter)
            if not prefer_dedicated and any(not item['claimed'] and not item['event'].is_set() and (shared_worker or item['job']['node_id']==node_id) and (not item['job'].get('packed') or packed_capable) for item in _pending.values()):
                event.set()
        try:
            await asyncio.wait_for(event.wait(), timeout=min(0.05, max(0.001, deadline-now)))
        except asyncio.TimeoutError:
            pass
        finally:
            with _lock:
                _waiters.discard(waiter)


async def _ply(request):
    # Only active, claimed jobs can be fetched; no arbitrary filesystem access.
    jid = request.match_info["job_id"]
    with _lock:
        item = _pending.get(jid)
        if item is None or not item["claimed"] or item["event"].is_set():
            raise web.HTTPNotFound(headers={"Cache-Control": "no-store"})
        payload = item["ply"]
    # The bytes are immutable. Keep a local reference while aiohttp sends the body.
    return web.Response(body=payload, content_type="application/octet-stream",
                        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


async def _complete(request):
    data = await request.json()
    jid = str(data.get("id", ""))
    with _lock:
        item = _pending.get(jid)
        if item is None or item["event"].is_set():
            return web.json_response({"ok": False, "error": "Unknown/expired job"}, status=404)
        try:
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            raw = base64.b64decode(data["png"], validate=True)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid PNG")
            item["result"] = raw
            item["browser_timings"] = _clean_timings(data.get("timings"))
        except Exception as exc:
            item["error"] = str(exc)
        finally:
            item["completed_at"] = time.perf_counter()
            item["event"].set()
    return web.json_response({"ok": True})


_BROWSER_STAGES = ("load", "ready", "compose", "pixels", "fetch", "parse", "fast", "direct", "shared", "packed")


def _clean_timings(data):
    if not isinstance(data, dict):
        return {}
    return {key: float(data[key]) for key in _BROWSER_STAGES
            if type(data.get(key)) in (int, float) and math.isfinite(data[key])
            and 0 <= data[key] <= 3600}


async def _complete_rgba(request):
    # An opaque sRGB canvas produces the same 8-bit RGB samples as lossless PNG.
    jid = request.match_info["job_id"]
    with _lock:
        item = _pending.get(jid)
        if item is None or not item["claimed"] or item["event"].is_set():
            return web.json_response({"ok": False, "error": "Unknown/expired job"}, status=404)
        size = item["job"].get("rgba_size")
    if not size or request.content_type != "application/octet-stream":
        return web.json_response({"ok": False, "error": "Raw pixels unavailable"}, status=400)
    expected = size[0] * size[1] * 4
    if request.content_length != expected:
        return web.json_response({"ok": False, "error": "Unexpected RGBA length"}, status=400)
    # No lock held during network I/O; the sender can fall back to PNG on errors.
    raw = await request.read()
    if len(raw) != expected:
        return web.json_response({"ok": False, "error": "Incomplete RGBA frame"}, status=400)
    try:
        timings = _clean_timings(json.loads(request.headers.get("X-Sharp-Timings", "{}")))
    except (ValueError, TypeError, OverflowError):
        timings = {}
    with _lock:
        if _pending.get(jid) is not item or item["event"].is_set():
            return web.json_response({"ok": False, "error": "Unknown/expired job"}, status=404)
        item["result"] = (raw, size[0], size[1])
        item["browser_timings"] = timings
        item["completed_at"] = time.perf_counter()
        item["event"].set()
    return web.json_response({"ok": True})


def packed_browser_available():
    if os.environ.get('SHARP_VR_BROWSER_LEGACY','0') == '1' or os.environ.get('SHARP_VR_PACKED','1') == '0':
        return False
    with _lock:
        now = time.monotonic()
        dedicated = _dedicated_seen is not None and now-_dedicated_seen < 10
        return now-_packed_seen.get(dedicated, -1e9) < 30


def submit_capture(ply_bytes, extrinsics, intrinsics, node_id, timeout=180, stereo=None,
                   expected_size=None):
    jid = uuid.uuid4().hex
    packed = isinstance(ply_bytes, bytes) and ply_bytes.startswith(b"SVRPACK1")
    if not isinstance(ply_bytes, bytes) or not (ply_bytes.startswith(b"ply\n") or packed):
        raise ValueError("Expected binary PLY bytes")
    url = f"/sharp_vr_video_batch/ply/{jid}.ply"
    job = {"id": jid, "node_id": str(node_id), "url": url, "filename": f"{jid}.ply",
           "extrinsics": extrinsics, "intrinsics": intrinsics, "timeout": timeout,
           "stereo": stereo, "packed": packed, "browser_fast": os.environ.get("SHARP_VR_BROWSER_LEGACY", "0") != "1"}
    # Bound raw transfer size; very large frames retain the previous PNG path.
    if expected_size and 0 < expected_size[0] * expected_size[1] * 4 <= 32 * 1024 * 1024:
        job["rgba_size"] = list(expected_size)
    item = {"event": threading.Event(), "result": None, "error": None,
            "job": job, "claimed": False, "ply": ply_bytes}
    item["submitted_at"] = time.perf_counter()
    item["deadline"] = time.monotonic() + timeout
    with _lock:
        _pending[jid] = item
        waiters = tuple(_waiters)
    for loop, event in waiters:
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            pass
    return jid, item


def cancel_capture(ticket):
    jid, item = ticket
    with _lock:
        if _pending.get(jid) is item:
            _pending.pop(jid)


def finish_capture(ticket, browser_timings=None):
    import comfy.model_management
    jid, item = ticket
    try:
        deadline = item["deadline"]
        comfy.model_management.throw_exception_if_processing_interrupted()
        while not item["event"].is_set():
            comfy.model_management.throw_exception_if_processing_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for SHARP shared renderer. Keep a ComfyUI page or the dedicated SHARP renderer page open with WebGL enabled. Background browser throttling may require keeping the renderer window visible.")
            item["event"].wait(min(remaining, 0.25))
        if item["completed_at"] - item["submitted_at"] > item["job"]["timeout"]:
            raise TimeoutError("SHARP renderer exceeded the frame timeout")
        if item["error"]:
            raise RuntimeError(item["error"])
        if browser_timings is not None:
            browser_timings.update(item.get("browser_timings", {}))
            browser_timings["roundtrip"] = item["completed_at"] - item["submitted_at"]
        return item["result"]
    finally:
        cancel_capture(ticket)


def capture(ply_bytes, extrinsics, intrinsics, node_id, timeout=180, stereo=None,
            expected_size=None, browser_timings=None):
    ticket = submit_capture(ply_bytes, extrinsics, intrinsics, node_id, timeout, stereo, expected_size)
    return finish_capture(ticket, browser_timings)
