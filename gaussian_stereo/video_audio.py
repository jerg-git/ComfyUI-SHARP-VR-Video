"""Optional local-video audio extraction. No imports from VHS and no lazy VHS AUDIO access."""
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess

log = logging.getLogger(__name__)
UPLOAD_TYPES = {"VHS_LoadVideo", "VHS_LoadVideoFFmpeg"}
PATH_TYPES = {"VHS_LoadVideoPath", "VHS_LoadVideoFFmpegPath"}
FFMPEG_TYPES = {"VHS_LoadVideoFFmpeg", "VHS_LoadVideoFFmpegPath"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".gif", ".avi", ".m4v", ".mpeg", ".mpg", ".ts", ".m2ts"}


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Expected a positive finite number")
    return result


def _link(value, slot):
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1] == slot:
        return str(value[0])
    return None


def _source(prompt, unique_id, video_info, count, fps):
    """Use only a directly connected, recognized loader; don't guess linked widgets."""
    import folder_paths
    inputs = prompt[str(unique_id)]["inputs"]
    source_id = _link(inputs.get("video_info"), 3)
    if source_id is None:
        raise ValueError("Connect video_info directly from a supported VHS loader")
    source = prompt[source_id]
    kind = source["class_type"]
    if kind not in UPLOAD_TYPES | PATH_TYPES:
        raise ValueError("Unsupported video_info source")
    settings = source["inputs"]
    # Meta-batching has a different timeline and VHS muxes audio only at finalization.
    if settings.get("meta_batch") is not None:
        raise ValueError("Automatic audio is not supported for VHS Meta Batch; supply full audio separately")
    if int(video_info["loaded_frame_count"]) != count:
        raise ValueError("Image count differs from video_info; audio timing is ambiguous")
    filename = settings.get("video")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Linked or missing video filename; supply audio separately")
    filename = filename.strip().strip('"')
    if "://" in filename or filename.startswith(("\\\\", "//")):
        raise ValueError("Automatic audio supports local video files only")
    if kind in UPLOAD_TYPES:
        root = Path(folder_paths.get_input_directory()).resolve()
        path = Path(folder_paths.get_annotated_filepath(filename)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Uploaded video is outside the input directory")
    else:
        path = Path(filename).resolve(strict=True)
        if "VHS_STRICT_PATHS" in os.environ and not path.is_relative_to(Path.cwd().resolve()):
            raise ValueError("Video is outside the permitted VHS path root")
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported local video file")
    if kind in FFMPEG_TYPES:
        start = float(settings.get("start_time", 0))
    else:
        # VHS applies skip BEFORE select_every_nth. loaded_fps already includes
        # the latter, so recover the pre-selection frame rate from metadata.
        nth = _positive(settings.get("select_every_nth", 1))
        start = float(settings.get("skip_first_frames", 0)) / (fps * nth)
    duration = count / fps
    if not math.isfinite(start) or start < 0:
        raise ValueError("Invalid video start time")
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
        raise ValueError("Invalid stereo PCM length")
    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)
    return {"waveform": waveform.reshape(-1, 2).T.unsqueeze(0).contiguous(),
            "sample_rate": 48000}


def resolve_video_audio(video_info, audio, prompt, unique_id, count, fallback_fps):
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
    if audio is not None:
        try:
            wave = audio["waveform"]
            rate = int(_positive(audio["sample_rate"]))
            if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[-1] == 0:
                raise ValueError("Invalid AUDIO waveform")
            return {"waveform": wave.detach().cpu().float().contiguous(), "sample_rate": rate}, fps
        except Exception:
            log.warning("[SHARP VR Video] Explicit audio unavailable; trying automatic audio")
    if video_info is not None:
        try:
            path, start, duration = _source(prompt or {}, unique_id, video_info, count, fps)
            return extract_audio(path, start, duration), fps
        except Exception as exc:
            # Do not log the source path or full FFmpeg stderr.
            log.warning("[SHARP VR Video] Automatic audio unavailable (%s); continuing without audio", type(exc).__name__)
    return None, fps
