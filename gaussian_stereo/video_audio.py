"""Optional local-video audio extraction. No imports from VHS and no lazy VHS AUDIO access."""
import logging
import math
import os
import re
from pathlib import Path
import shutil
import subprocess

log = logging.getLogger(__name__)
UPLOAD_TYPES = {"VHS_LoadVideo", "VHS_LoadVideoFFmpeg"}
PATH_TYPES = {"VHS_LoadVideoPath", "VHS_LoadVideoFFmpegPath"}
FFMPEG_TYPES = {"VHS_LoadVideoFFmpeg", "VHS_LoadVideoFFmpegPath"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".gif", ".avi", ".m4v", ".mpeg", ".mpg", ".ts", ".m2ts"}


class AudioSourceError(ValueError):
    """Diagnostic with a fixed, path-free message safe to log."""


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise AudioSourceError("Expected a positive finite number")
    return result


def _link(value, slot):
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1] == slot:
        return str(value[0])
    return None


def _selection(prompt, value, depth=0):
    """Resolve literal selectors and known integer/selected-index nodes only."""
    if depth > 16:
        raise AudioSourceError("Switch selection chain is too long")
    if type(value) is int and value >= 1:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        node = prompt.get(str(value[0]), {})
        kind, settings = node.get("class_type"), node.get("inputs", {})
        if kind in {"ImpactInt", "PrimitiveInt"} and value[1] == 0:
            return _selection(prompt, settings.get("value"), depth+1)
        if kind == "ImpactSwitch" and value[1] == 2:
            return _selection(prompt, settings.get("select"), depth+1)
    raise AudioSourceError("Use a literal switch selection or ImpactInt; dynamic selections are unsupported")


def trace_loader(prompt, unique_id, input_name="video_info", output_slot=3):
    """Trace selected ImpactSwitch value outputs, never inspect unselected branches."""
    current = prompt.get(str(unique_id), {}).get("inputs", {}).get(input_name)
    visited = set()
    while isinstance(current, (list, tuple)) and len(current) == 2:
        node_id, slot = str(current[0]), current[1]
        if node_id in visited or len(visited) >= 32:
            raise AudioSourceError("Cycle or excessive depth in video source connections")
        visited.add(node_id)
        node = prompt.get(node_id, {})
        kind, settings = node.get("class_type"), node.get("inputs", {})
        if kind in UPLOAD_TYPES | PATH_TYPES and slot == output_slot:
            return node_id, kind, settings
        if kind == "ImpactSwitch" and slot == 0:
            selected = _selection(prompt, settings.get("select"))
            current = settings.get(f"input{selected}")
            continue
        raise AudioSourceError(f"Unsupported {input_name} source; use a VHS loader or Impact Switch (Any)")
    raise AudioSourceError(f"No selected VHS loader connected to {input_name}")


def _filename(prompt, value):
    """Resolve the literal path field in the bundled RES4LYF workflow.

    Do not execute arbitrary graph nodes or guess dynamic string values.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1] == 0:
        node = prompt.get(str(value[0]), {})
        settings = node.get("inputs", {})
        if node.get("class_type") == "TextBoxConcatenate":
            if settings.get("text_external") in (None, "") and settings.get("separator", "") == "" and settings.get("mode", "prepend_external_input") in ("prepend_external_input", "append_external_input"):
                value = settings.get("text")
    if not isinstance(value, str) or not value.strip():
        raise AudioSourceError("Unsupported linked or missing video filename; use a literal path or the sample TextBoxConcatenate field")
    return value


def _manager_id(meta_batch, prompt, unique_id):
    node_id = _link(prompt.get(str(unique_id), {}).get("inputs", {}).get("meta_batch"), 0)
    if node_id is None or prompt.get(node_id, {}).get("class_type") != "VHS_BatchManager":
        raise AudioSourceError("Connect the same VHS Meta Batch Manager directly to SHARP")
    runtime_id = getattr(meta_batch, "unique_id", None)
    if runtime_id is not None and str(runtime_id) != node_id:
        raise AudioSourceError("Runtime and graph Meta Batch Managers differ")
    return node_id


def _encoder_active(meta_batch, prompt, unique_id):
    ids = [str(nid) for nid, node in prompt.items()
           if node.get("class_type") == "VHS_VideoCombine"
           and _link(node.get("inputs", {}).get("images"), 0) == str(unique_id)]
    return any(str(key) in ids for key in meta_batch.outputs) if ids else bool(meta_batch.outputs)


def _check_meta_format(settings):
    # These built-in VHS formats do not truncate frame counts to a modulus.
    # Other model-oriented formats can discard trailing frames.
    if settings.get("format", "None") not in (None, "None", "AnimateDiff"):
        raise AudioSourceError("Use VHS loader format None or AnimateDiff for Meta Batch timing")


def meta_timeline(meta_batch, prompt, unique_id, video_info, count):
    """Track actual chunk offsets against VHS encoder lifetime.

    The final chunk can be short. Source frame count is an estimate
    when resampling or reading variable-rate video.
    """
    if meta_batch is None:
        return 0, count, True
    prompt = prompt or {}
    node_id = _manager_id(meta_batch, prompt, unique_id)
    manager = prompt.get(node_id, {})
    if manager.get("class_type") != "VHS_BatchManager":
        raise AudioSourceError("Unsupported Meta Batch Manager")
    requeue = manager.get("inputs", {}).get("requeue", 0)
    if type(requeue) is not int or requeue < 0:
        raise AudioSourceError("Invalid VHS requeue counter")
    size = int(meta_batch.frames_per_batch)
    final = bool(meta_batch.has_closed_inputs)
    if size <= 0 or count > size or (not final and count != size):
        raise AudioSourceError("Meta Batch requires unmodified, full-size intermediate chunks")
    # The manager's total_frames estimate ignores trim in some VHS versions.
    # Derive the selected clip length from the active loader instead.
    source_id, kind, settings = trace_loader(prompt, unique_id)
    if _link(settings.get("meta_batch"), 0) != node_id:
        raise AudioSourceError("Selected VHS loader uses a different Meta Batch Manager")
    _check_meta_format(settings)
    # VHS can carry a nonzero requeue value into a fresh run after reset.
    # Anchor to the actual encoder lifetime instead of assuming requeue=0.
    # Store only small counters on the manager, never frames or audio.
    state = getattr(meta_batch, "_sharp_vr_timeline", None)
    if state is None:
        state = {}
        meta_batch._sharp_vr_timeline = state
    key = str(unique_id)
    previous = state.get(key)
    active = _encoder_active(meta_batch, prompt, unique_id)
    if not active:
        offset = 0
        if requeue != 0:
            log.warning("[SHARP VR Video] Fresh encoder with VHS requeue=%d; starting SHARP timeline at frame 1", requeue)
    elif previous is not None and previous[0] == source_id and requeue == previous[1] + 1:
        offset = previous[2] + previous[3]
    else:
        raise AudioSourceError("Cannot verify continuity with existing VHS encoder; restart the queue")
    state[key] = (source_id, requeue, offset, count)
    if final:
        state.pop(key, None)
        return offset, offset + count, True
    fps = _positive(video_info["loaded_fps"])
    source_fps = _positive(video_info["source_fps"])
    if kind in FFMPEG_TYPES:
        remaining = max(0.0, float(video_info["source_duration"]) - float(settings.get("start_time", 0)))
        estimate = round(remaining * fps)
    else:
        nth = int(_positive(settings.get("select_every_nth", 1)))
        raw_count = _positive(video_info["source_frame_count"])
        rate = float(settings.get("force_rate", 0)) or source_fps
        before_skip = math.ceil(raw_count * rate / source_fps - 1e-8)
        estimate = math.ceil(max(0, before_skip - int(settings.get("skip_first_frames", 0))) / nth)
    cap = int(settings.get("frame_load_cap", 0))
    if cap > 0:
        estimate = min(cap, estimate)
    return offset, max(offset + count + 1, estimate), False


def _source(prompt, unique_id, video_info, count, fps, meta_batch=None, total_count=None):
    """Resolve direct or selected VHS loader, with fail-safe timing checks."""
    import folder_paths
    source_id, kind, settings = trace_loader(prompt, unique_id)
    # Catch mismatched switches when the IMAGE branch is also traceable.
    try:
        image_source, _, _ = trace_loader(prompt, unique_id, "image", 0)
    except AudioSourceError:
        image_source = None
    if image_source is not None and image_source != source_id:
        raise AudioSourceError("IMAGE and video_info switches select different VHS loaders")
    if settings.get("meta_batch") is not None:
        if meta_batch is None or _link(settings.get("meta_batch"), 0) != _manager_id(meta_batch, prompt, unique_id):
            raise AudioSourceError("Connect the selected loader's Meta Batch Manager to SHARP")
        _check_meta_format(settings)
        if total_count is None or not meta_batch.has_closed_inputs:
            raise AudioSourceError("Full clip audio is available only on the final Meta Batch")
    elif meta_batch is not None:
        raise AudioSourceError("Selected VHS loader has no Meta Batch Manager connected")
    if int(video_info["loaded_frame_count"]) != count:
        raise AudioSourceError(f"Image count ({count}) differs from video_info ({int(video_info['loaded_frame_count'])}); audio timing is ambiguous")
    filename = _filename(prompt, settings.get("video"))
    filename = filename.strip().strip('"')
    if ("://" in filename and not re.match(r"^[A-Za-z]:[/\\]", filename)) or filename.startswith(("\\\\", "//")):
        raise AudioSourceError("Automatic audio supports local video files only")
    if kind in UPLOAD_TYPES:
        root = Path(folder_paths.get_input_directory()).resolve()
        path = Path(folder_paths.get_annotated_filepath(filename)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise AudioSourceError("Uploaded video is outside the input directory")
    else:
        path = Path(filename).resolve(strict=True)
        if "VHS_STRICT_PATHS" in os.environ and not path.is_relative_to(Path.cwd().resolve()):
            raise AudioSourceError("Video is outside the permitted VHS path root")
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise AudioSourceError("Unsupported local video file")
    if kind in FFMPEG_TYPES:
        start = float(settings.get("start_time", 0))
    else:
        # VHS applies skip BEFORE select_every_nth. loaded_fps already includes
        # the latter, so recover the pre-selection frame rate from metadata.
        nth = _positive(settings.get("select_every_nth", 1))
        start = float(settings.get("skip_first_frames", 0)) / (fps * nth)
    duration = (total_count if total_count is not None else count) / fps
    if not math.isfinite(start) or start < 0:
        raise AudioSourceError("Invalid video start time")
    return path, start, duration


def _ffmpeg():
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced:
        return forced
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return shutil.which("ffmpeg")


def extract_audio(path, start, duration):
    """Decode the first audio stream to eager ComfyUI AUDIO or return None.

    Use optional stream mapping and check empty output. Stereo 48 kHz float
    PCM is deliberate: no parsing FFmpeg's human-readable channel metadata.
    Only local file/pipe protocols are permitted; no temporary files.
    """
    import torch
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    args = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-ss", f"{start:.9f}",
            "-i", str(path), "-t", f"{duration:.9f}", "-map", "0:a:0?",
            "-vn", "-sn", "-dn", "-ac", "2", "-ar", "48000",
            "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, capture_output=True, timeout=300,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        # No audio stream is normal. Other failures are reported but do not
        # abort the stereo batch (handled by resolve_video_audio).
        err = result.stderr.decode("utf-8", "replace")
        if "does not contain any stream" in err or "matches no streams" in err:
            return None
        raise RuntimeError("FFmpeg could not decode the audio stream")
    if not result.stdout:
        return None
    if len(result.stdout) % 8:
        raise AudioSourceError("Invalid stereo PCM length")
    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)
    return {"waveform": waveform.reshape(-1, 2).T.unsqueeze(0).contiguous(),
            "sample_rate": 48000}


def resolve_video_audio(video_info, audio, prompt, unique_id, count, fallback_fps, meta_batch=None, total_count=None):
    """Explicit standard AUDIO wins; otherwise optional automatic VHS audio."""
    fps = 24.0
    try:
        fps = _positive(fallback_fps)
    except (TypeError, ValueError, OverflowError):
        log.warning("[SHARP VR Video] Invalid fallback FPS; using 24")
    if video_info is not None:
        try:
            fps = _positive(video_info["loaded_fps"])
        except (KeyError, TypeError, ValueError, OverflowError):
            log.warning("[SHARP VR Video] Invalid video_info FPS; using fallback")
    if meta_batch is not None and not meta_batch.has_closed_inputs:
        return None, fps
    if audio is not None:
        try:
            wave = audio["waveform"]
            rate = int(_positive(audio["sample_rate"]))
            if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[-1] == 0:
                raise AudioSourceError("Invalid AUDIO waveform")
            return {"waveform": wave.detach().cpu().float().contiguous(), "sample_rate": rate}, fps
        except Exception:
            log.warning("[SHARP VR Video] Explicit audio unavailable; trying automatic audio")
    if video_info is not None:
        try:
            path, start, duration = _source(prompt or {}, unique_id, video_info, count, fps, meta_batch, total_count)
            return extract_audio(path, start, duration), fps
        except AudioSourceError as exc:
            log.warning("[SHARP VR Video] Automatic audio unavailable: %s; continuing without audio", exc)
        except Exception as exc:
            # Do not log the source path or full FFmpeg stderr.
            log.warning("[SHARP VR Video] Automatic audio unavailable (%s); continuing without audio", type(exc).__name__)
    return None, fps
