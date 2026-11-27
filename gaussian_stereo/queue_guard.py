"""Automatic QuietNoise Queue Manager priority repair for SHARP Meta Batch workflows.

This patch is intentionally narrow:
- only activates when QuietNoise Queue Manager is detected
- only reorders recognized SHARP+VHS Meta Batch continuations
- never rewrites Queue Manager's database rows
- leaves ordinary native queue behavior untouched
"""
from __future__ import annotations

import functools
import json
import logging
import threading
from typing import Any

LOG = logging.getLogger(__name__)
MARKER = "_sharp_vr_meta_batch_queue_guard_v1"
SHARP_NODE_ID = "SharpVRVideoBatch"


def _parts(item: Any):
    if not isinstance(item, (tuple, list)) or len(item) not in (5, 6):
        return None
    if not isinstance(item[2], dict) or not isinstance(item[4], (list, tuple, set)):
        return None
    return item[2], {str(x) for x in item[4]}


def _link(value, index=0):
    return value[index] if isinstance(value, (list, tuple)) and len(value) > index else None


def _guarded_managers(item):
    """Return VHS manager ids for SHARP workflows that are safe to protect."""
    parts = _parts(item)
    if parts is None:
        return set()
    prompt, _targets = parts

    sharp_nodes = {}
    for uid, node in prompt.items():
        if node.get("class_type") != SHARP_NODE_ID:
            continue
        manager = _link(node.get("inputs", {}).get("meta_batch"), 0)
        if manager is None:
            continue
        manager = str(manager)
        if prompt.get(manager, {}).get("class_type") != "VHS_BatchManager":
            continue
        sharp_nodes[str(uid)] = manager

    if not sharp_nodes:
        return set()

    result = set()
    for node in prompt.values():
        if node.get("class_type") != "VHS_VideoCombine":
            continue
        inputs = node.get("inputs", {})
        sharp_uid = _link(inputs.get("images"), 0)
        manager = _link(inputs.get("meta_batch"), 0)
        if sharp_uid is None or manager is None:
            continue
        sharp_uid = str(sharp_uid)
        manager = str(manager)
        if sharp_nodes.get(sharp_uid) == manager:
            result.add(manager)
    return result


def _normalized(prompt):
    result = {}
    for uid, node in prompt.items():
        inputs = dict(node.get("inputs", {}))
        if node.get("class_type") == "VHS_BatchManager":
            inputs.pop("requeue", None)
        result[uid] = (node.get("class_type"), inputs)
    return result


def _is_continuation(item, running):
    parts = _parts(item)
    previous = _parts(running)
    if parts is None or previous is None or item[1] == running[1]:
        return False
    if not isinstance(item[0], (int, float)) or item[0] >= 0:
        return False

    managers = _guarded_managers(item)
    if not managers or managers != _guarded_managers(running):
        return False

    prompt, targets = parts
    old, old_targets = previous
    if targets != old_targets:
        return False

    for uid in managers:
        count = prompt[uid]["inputs"].get("requeue", 0)
        before = old[uid]["inputs"].get("requeue", 0)
        # VHS shallow-copies the prompt and can mutate the running counter too.
        if type(count) is not int or type(before) is not int or count <= 0 or count not in (before, before + 1):
            return False
    return _normalized(prompt) == _normalized(old)


def _pending_row(read_single, prompt_id):
    return read_single(
        "SELECT number, prompt, updated_at FROM queue WHERE prompt_id = ? AND status = 0",
        (prompt_id,),
    )


def _candidate(owner, row, incoming):
    if callable(getattr(owner, "_pending_prompt", None)):
        candidate = owner._pending_prompt(row)
    else:
        # Older Queue Manager revisions reconstruct the tuple from stored JSON.
        saved = json.loads(row[1])
        saved[0] = row[0]
        if len(incoming) == 6 and len(saved) == 5:
            saved.append(incoming[5])
        candidate = tuple(saved)
    if _parts(candidate) is None or candidate[1] != incoming[1] or len(candidate) != len(incoming):
        raise ValueError("Unexpected stored continuation format")
    return candidate


def install(native_queue):
    """Wrap only the recognized QuietNoise Queue Manager instance, once."""
    original = native_queue.put
    if getattr(original, MARKER, False):
        return True

    owner = getattr(original, "__self__", None)
    method = getattr(original, "__func__", None)
    if (
        owner is None
        or method is None
        or type(owner).__name__ != "QM_Queue"
        or not type(owner).__module__.endswith(".qm_queue")
        or method.__name__ != "queue_put"
        or getattr(owner, "native_queue", None) is not native_queue
        or not isinstance(native_queue.mutex, type(threading.RLock()))
    ):
        return False

    read_single = method.__globals__.get("read_single")
    if not callable(read_single):
        return False

    @functools.wraps(original)
    def put(item):
        with native_queue.mutex:
            protect = False
            try:
                running = list(native_queue.currently_running.values())
                protect = len(running) == 1 and _is_continuation(item, running[0])
            except Exception as exc:
                LOG.warning("[SHARP VR Video] Queue guard recognition skipped: %s", type(exc).__name__)

            result = original(item)
            if not protect or owner.paused:
                return result

            try:
                pending = native_queue.queue
                if any(entry[1] == item[1] for entry in pending):
                    return result
                if len(pending) > 1:
                    LOG.warning("[SHARP VR Video] Queue guard saw multiple native entries; leaving queue unchanged")
                    return result

                row = _pending_row(read_single, item[1])
                if row is None:
                    LOG.warning("[SHARP VR Video] Queue guard could not find continuation in pending database; leaving queue unchanged")
                    return result

                if pending and _pending_row(read_single, pending[0][1]) is None:
                    LOG.warning("[SHARP VR Video] Queue guard saw a non-database native entry; leaving queue unchanged")
                    return result

                candidate = _candidate(owner, row, item)
                native_queue.queue[:] = [candidate]
                native_queue.not_empty.notify()
                LOG.info("[SHARP VR Video] Meta Batch continuation promoted ahead of prefetched Queue Manager job")
            except Exception as exc:
                LOG.warning("[SHARP VR Video] Queue guard priority repair unavailable: %s", type(exc).__name__)
            return result

    setattr(put, MARKER, True)
    native_queue.put = put
    LOG.info("[SHARP VR Video] QuietNoise Queue Manager Meta Batch guard installed")
    return True


def try_install():
    """Best-effort install. Quietly returns False when Queue Manager is absent."""
    try:
        from server import PromptServer
        native_queue = PromptServer.instance.prompt_queue
    except Exception:
        return False

    try:
        return install(native_queue)
    except Exception as exc:
        LOG.warning("[SHARP VR Video] Queue guard install failed: %s", type(exc).__name__)
        return False
