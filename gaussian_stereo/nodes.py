"""Modified ComfyUI node integration based on ComfyUI-Sharp (GPL-3.0).

ComfyUI node backend for the standalone Gaussian Stereo Preview extension.

The camera inputs retain the established custom socket type names
EXTRINSICS and INTRINSICS for workflow compatibility. They are not STRING inputs.
"""

import math
from pathlib import Path
import threading

import folder_paths


_TEMP_PLY_BY_NODE = {}
_TEMP_PLY_LOCK = threading.Lock()


class GaussianStereoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to a Gaussian Splatting PLY file inside ComfyUI input, output, or temp.",
                }),
            },
            "optional": {
                # Keep the established camera socket identifiers for compatibility.
                "extrinsics": ("EXTRINSICS",),
                "intrinsics": ("INTRINSICS",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "utils/preview"
    OUTPUT_NODE = True

    @staticmethod
    def _allowed_roots():
        roots = []
        for getter in (
            folder_paths.get_input_directory,
            folder_paths.get_output_directory,
            folder_paths.get_temp_directory,
        ):
            try:
                roots.append(Path(getter()).resolve())
            except Exception:
                pass
        return roots

    @classmethod
    def _resolve_ply_path(cls, raw_path):
        value = str(raw_path or "").strip().strip('"')
        if not value:
            raise ValueError("PLY path is empty")
        candidate = Path(value).expanduser()
        roots = cls._allowed_roots()
        candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
        for item in candidates:
            try:
                path = item.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (path.is_file() and path.suffix.lower() == ".ply"
                    and any(path == root or root in path.parents for root in roots)):
                return path
        raise ValueError(
            "PLY file was not found inside ComfyUI input, output, or temp directories: "
            f"{value}"
        )

    @staticmethod
    def _to_builtin(value):
        """Convert common matrix containers (numpy arrays/tuples) to JSON-like lists."""
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [GaussianStereoPreview._to_builtin(item) for item in value]
        # Convert numpy and other numeric scalar types without importing numpy.
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Camera matrices must contain only finite numbers")
            return number
        raise ValueError(f"Unsupported camera matrix value: {type(value).__name__}")

    @classmethod
    def _validate_matrix(cls, value, name, rows, cols):
        if value is None:
            return None
        matrix = cls._to_builtin(value)
        if (not isinstance(matrix, list) or len(matrix) != rows or
                any(not isinstance(row, list) or len(row) != cols for row in matrix)):
            raise ValueError(f"{name} must be a {rows}x{cols} matrix")
        return matrix

    @classmethod
    def IS_CHANGED(cls, ply_path, extrinsics=None, intrinsics=None):
        try:
            path = cls._resolve_ply_path(ply_path)
            stat = path.stat()
            # Include camera values so changes to connected upstream nodes trigger refresh.
            return repr((path, stat.st_mtime_ns, stat.st_size, extrinsics, intrinsics))
        except Exception:
            return float("nan")

    def preview(self, ply_path, extrinsics=None, intrinsics=None):
        try:
            path = self._resolve_ply_path(ply_path)
            extrinsics_data = self._validate_matrix(extrinsics, "extrinsics", 4, 4)
            intrinsics_data = self._validate_matrix(intrinsics, "intrinsics", 3, 3)
            size_mb = path.stat().st_size / (1024 * 1024)

            ui_data = {
                "ply_file": [str(path)],
                "filename": [path.name],
                "file_size_mb": [round(size_mb, 2)],
            }
            if extrinsics_data is not None:
                ui_data["extrinsics"] = [extrinsics_data]
            if intrinsics_data is not None:
                ui_data["intrinsics"] = [intrinsics_data]
            return {"ui": ui_data, "result": ()}
        except Exception as exc:
            return {"ui": {"error": [str(exc)]}, "result": ()}


class SharpVRVideoPreview(GaussianStereoPreview):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHARP_VR_VIDEO_MODEL_CONFIG",),
                "image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 500.0, "step": 0.1,
                    "tooltip": (
                        "SHARP camera focal length in millimeters. Changes the "
                        "reconstructed geometry and requires a new inference. "
                        "Perceived Depth Scale provides a real-time viewer-side alternative."
                    ),
                }),
                "save_to_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS",),
                "intrinsics": ("INTRINSICS",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "utils/preview"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model, image, focal_length_mm=30.0,
                   save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        return float("nan")

    def run(self, model, image, focal_length_mm=30.0,
            save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        try:
            if getattr(image, "ndim", None) != 4:
                raise ValueError("Expected IMAGE tensor with shape [B,H,W,C]")
            if image.shape[0] != 1:
                raise ValueError("SHARP VR Video Preview currently supports one image per run")
            output_dir = (folder_paths.get_output_directory() if save_to_output
                          else folder_paths.get_temp_directory())
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            from .sharp_inference import run_sharp_inference
            ply_value, sharp_extrinsics, sharp_intrinsics = run_sharp_inference(
                model_config=model, image=image, output_dir=output_dir,
                focal_length_mm=focal_length_mm, extrinsics=extrinsics, intrinsics=intrinsics,
            )
            path = self._resolve_ply_path(ply_value)
            if not save_to_output and unique_id is not None:
                node_key = str(unique_id)
                temp_root = Path(folder_paths.get_temp_directory()).resolve()
                with _TEMP_PLY_LOCK:
                    previous = _TEMP_PLY_BY_NODE.get(node_key)
                    if previous is not None and previous != path:
                        try:
                            previous_resolved = Path(previous).resolve()
                            if (previous_resolved.parent == temp_root
                                    and previous_resolved.name.startswith("sharp_vr_video_")
                                    and previous_resolved.suffix.lower() == ".ply"):
                                previous_resolved.unlink(missing_ok=True)
                        except OSError as cleanup_error:
                            print(f"[Gaussian Stereo] Could not remove previous temp PLY {previous}: {cleanup_error}")
                    _TEMP_PLY_BY_NODE[node_key] = path
            extrinsics_data = self._validate_matrix(sharp_extrinsics, "extrinsics", 4, 4)
            intrinsics_data = self._validate_matrix(sharp_intrinsics, "intrinsics", 3, 3)
            ui_data = {
                "ply_file": [str(path)],
                "filename": [path.name],
                "file_size_mb": [round(path.stat().st_size / (1024 * 1024), 2)],
                "save_to_output": [bool(save_to_output)],
            }
            if extrinsics_data is not None:
                ui_data["extrinsics"] = [extrinsics_data]
            if intrinsics_data is not None:
                ui_data["intrinsics"] = [intrinsics_data]
            return {"ui": ui_data, "result": ()}
        except Exception as exc:
            return {"ui": {"error": [str(exc)]}, "result": ()}


# Preset names are frontend-only convenience labels. The backend always
# renders the actual widget values, never overrides them from the preset.
STEREO_PRESETS = [
    "Legacy / Viewer Controls",
    "VR Life-size / Half Body",
    "VR Life-size / Mixed Shots",
    "VR Life-size / Full Body",
    "VR 3D Cinema / Comfortable",
    "VR 3D Cinema / Immersive",
    "Custom / Fixed Stereo",
    "Cross-eyed / Freeview",
]
PROJECTION_MODES = ["Parallel off-axis", "Parallel", "Toe-in", "Legacy / Parallel", "Legacy / Toe-in"]


def resolve_stereo_settings(projection_mode, baseline_mm, convergence_m,
                            stereo_intensity, depth_scale, splat_size_scale,
                            camera_dolly_m=0.0, camera_orbit_yaw_deg=0.0,
                            camera_orbit_pitch_deg=0.0):
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unknown stereo projection: {projection_mode}")
    def bounded(value, low, high, name):
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value
    return {"mode": "relative" if projection_mode.startswith("Legacy / ") else "fixed",
            "projection": projection_mode,
            "baseline_m": bounded(baseline_mm, 0, 100, "Baseline (mm)") / 1000,
            "convergence_m": bounded(convergence_m, 0, 100, "Convergence (m)"),
            "intensity": bounded(stereo_intensity, -100, 100, "Relative intensity (%)"),
            "depth_scale": bounded(depth_scale, 0.1, 3, "Depth scale"),
            "splat_scale": bounded(splat_size_scale, 0.01, 100, "Splat size scale"),
            "dolly_m": bounded(camera_dolly_m, 0, 0.4, "Camera dolly (m)"),
            "orbit_yaw_deg": bounded(camera_orbit_yaw_deg, -10, 10, "Orbit yaw (degrees)"),
            "orbit_pitch_deg": bounded(camera_orbit_pitch_deg, -10, 10, "Orbit pitch (degrees)")}


CAMERA_MOTION_MODES = ["Fixed", "Animated"]
DOLLY_MOTION_PATTERNS = {
    "0 → Target": (0, 1),
    "Target → 0": (1, 0),
    "0 → Target → 0": (0, 1, 0),
    "Target → 0 → Target": (1, 0, 1),
}
ORBIT_MOTION_PATTERNS = {
    "0 → Target": (0, 1),
    "Target → 0": (1, 0),
    "−Target → Target": (-1, 1),
    "Target → −Target": (1, -1),
    "0 → Target → 0": (0, 1, 0),
    "Target → 0 → Target": (1, 0, 1),
    "−Target → Target → −Target": (-1, 1, -1),
    "Target → −Target → Target": (1, -1, 1),
}
DEFAULT_CAMERA_PATTERN = "0 → Target"


def camera_motion_keys(mode, pattern, patterns):
    """Validate active motion settings; ignore inactive pattern in Fixed mode."""
    if mode == "Fixed":
        return (1,)
    if mode != "Animated" or pattern not in patterns:
        raise ValueError(f"Invalid camera motion: {mode!r}, {pattern!r}")
    return patterns[pattern]


def camera_motion_value(target, keys, index, count):
    """Sample signed keyframes with cosine ease-in/out on each leg.

    The first and last frames are exact endpoints; a one-frame batch uses
    the first keyframe. Compute from the frame index, never accumulated deltas.
    """
    if len(keys) == 1 or count <= 1 or index == 0:
        return target * keys[0]
    if index == count - 1:
        return target * keys[-1]
    position = index * (len(keys) - 1) / (count - 1)
    leg = min(int(position), len(keys) - 2)
    fraction = position - leg
    eased = (1.0 - math.cos(math.pi * fraction)) * 0.5
    return target * (keys[leg] + (keys[leg + 1] - keys[leg]) * eased)



class SharpVRVideoBatch(GaussianStereoPreview):
    """Run SHARP per input frame and capture the original gsplat viewer's stereo canvas."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SHARP_VR_VIDEO_MODEL_CONFIG",), "image": ("IMAGE",),
            "focal_length_mm": ("FLOAT", {"default":35.0,"min":0.0,"max":500.0,"step":0.1}),
            "frame_timeout_seconds": ("INT", {"default":180,"min":30,"max":900,"step":10})},
            "optional": {
                "stereo_preset": ("COMBO", {"options": STEREO_PRESETS, "default": "VR Life-size / Mixed Shots",
                    "tooltip": "Selecting a preset fills the editable widgets once. The actual widget values are used for rendering."}),
                "custom_baseline_mm": ("FLOAT", {"default":60.0,"min":0.0,"max":100.0,"step":0.1,
                    "tooltip": "Virtual camera separation in millimeters, used in fixed projection modes."}),
                "custom_convergence_m": ("FLOAT", {"default":2.5,"min":0.0,"max":100.0,"step":0.1,
                    "tooltip": "Zero-parallax distance for off-axis or toe-in projection. 0 means infinity."}),
                "projection_mode": ("COMBO", {"options": PROJECTION_MODES, "default": "Parallel off-axis"}),
                "stereo_intensity": ("FLOAT", {"default":30.0,"min":-100.0,"max":100.0,"step":1.0,
                    "tooltip": "Relative stereo intensity (percent). Only used in the Legacy projection modes."}),
                "depth_scale": ("FLOAT", {"default":1.0,"min":0.1,"max":3.0,"step":0.01}),
                "splat_size_scale": ("FLOAT", {"default":1.0,"min":0.01,"max":100.0,"step":0.01}),
                "camera_dolly_m": ("FLOAT", {"default":0.0,"min":0.0,"max":0.4,"step":0.001,"round":0.001,
                    "tooltip": "Move both cameras closer in metres (like the preview wheel). 0 = original camera position."}),
                "camera_orbit_yaw_deg": ("FLOAT", {"default":0.0,"min":-10.0,"max":10.0,"step":0.001,"round":0.001,
                    "tooltip": "Orbit horizontally about the fixed convergence distance (1 m in legacy/infinite modes). Positive looks right."}),
                "camera_orbit_pitch_deg": ("FLOAT", {"default":0.0,"min":-10.0,"max":10.0,"step":0.001,"round":0.001,
                    "tooltip": "Orbit vertically about the same fixed pivot. Positive looks up."}),
                # Append new widgets; existing saved widget positions remain unchanged.
                "camera_dolly_mode": ("COMBO", {"options": CAMERA_MOTION_MODES, "default": "Fixed"}),
                "camera_dolly_pattern": ("COMBO", {"options": list(DOLLY_MOTION_PATTERNS), "default": DEFAULT_CAMERA_PATTERN,
                    "tooltip": "Animated dolly across this input batch. Target is camera_dolly_m."}),
                "camera_orbit_yaw_mode": ("COMBO", {"options": CAMERA_MOTION_MODES, "default": "Fixed"}),
                "camera_orbit_yaw_pattern": ("COMBO", {"options": list(ORBIT_MOTION_PATTERNS), "default": DEFAULT_CAMERA_PATTERN,
                    "tooltip": "Animated horizontal orbit. Target retains its sign; −Target negates it."}),
                "camera_orbit_pitch_mode": ("COMBO", {"options": CAMERA_MOTION_MODES, "default": "Fixed"}),
                "camera_orbit_pitch_pattern": ("COMBO", {"options": list(ORBIT_MOTION_PATTERNS), "default": DEFAULT_CAMERA_PATTERN,
                    "tooltip": "Animated vertical orbit. Target retains its sign; −Target negates it."}),
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Optional VHS loader video_info. Leave VHS audio disconnected."}),
                "audio": ("AUDIO", {"tooltip": "Optional explicit standard audio override. Do not connect the VHS loader's lazy audio output."}),
                "fallback_fps": ("FLOAT", {"default":24.0,"min":0.001,"max":240.0,"step":0.001,"round":0.001,
                    "tooltip": "Used when video_info is absent or has no valid loaded FPS."}),
                "meta_batch": ("VHS_BatchManager", {"tooltip": "Optional: same VHS Meta Batch Manager as loader and combiner; keeps animation continuous and supplies full audio on final chunk."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    OUTPUT_NODE = False
    RETURN_NAMES = ("stereo_frames", "audio", "fps")
    FUNCTION = "run_batch"
    CATEGORY = "utils/preview"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def run_batch(self, model, image, focal_length_mm=35.0, frame_timeout_seconds=180,
                  unique_id=None, stereo_preset="VR Life-size / Mixed Shots",
                  custom_baseline_mm=60.0, custom_convergence_m=2.5,
                  projection_mode="Parallel off-axis", stereo_intensity=30.0,
                  depth_scale=1.0, splat_size_scale=1.0,
                  camera_dolly_m=0.0, camera_orbit_yaw_deg=0.0,
                  camera_orbit_pitch_deg=0.0,
                  camera_dolly_mode="Fixed", camera_dolly_pattern=DEFAULT_CAMERA_PATTERN,
                  camera_orbit_yaw_mode="Fixed", camera_orbit_yaw_pattern=DEFAULT_CAMERA_PATTERN,
                  camera_orbit_pitch_mode="Fixed", camera_orbit_pitch_pattern=DEFAULT_CAMERA_PATTERN,
                  video_info=None, audio=None, fallback_fps=24.0, prompt=None, meta_batch=None):
        import io
        import time
        import numpy as np
        import torch
        import comfy.model_management
        from comfy.utils import ProgressBar
        from PIL import Image
        from tqdm import tqdm
        from .sharp_inference import video_inference_frames, clear_video_batch_cache
        from .batch_bridge import submit_capture, finish_capture, cancel_capture, packed_browser_available
        import os

        if getattr(image, "ndim", None) != 4 or image.shape[0] < 1 or image.shape[-1] != 3:
            raise ValueError("Expected RGB IMAGE batch [B,H,W,3]")
        if unique_id is None:
            raise ValueError("Batch node requires its ComfyUI unique_id")
        stereo = resolve_stereo_settings(projection_mode, custom_baseline_mm, custom_convergence_m,
                                         stereo_intensity, depth_scale, splat_size_scale,
                                         camera_dolly_m, camera_orbit_yaw_deg, camera_orbit_pitch_deg)
        count, height, width, _ = image.shape
        motion = {
            "dolly_m": camera_motion_keys(camera_dolly_mode, camera_dolly_pattern, DOLLY_MOTION_PATTERNS),
            "orbit_yaw_deg": camera_motion_keys(camera_orbit_yaw_mode, camera_orbit_yaw_pattern, ORBIT_MOTION_PATTERNS),
            "orbit_pitch_deg": camera_motion_keys(camera_orbit_pitch_mode, camera_orbit_pitch_pattern, ORBIT_MOTION_PATTERNS),
        }
        animated = {key: keys for key, keys in motion.items() if len(keys) > 1}
        from .video_audio import AudioSourceError, meta_timeline, resolve_video_audio
        import logging
        offset, timeline_count, final = 0, count, True
        timeline_valid = True
        if meta_batch is not None:
            try:
                offset, timeline_count, final = meta_timeline(meta_batch, prompt, unique_id, video_info, count)
                if video_info is not None and int(video_info["loaded_frame_count"]) != count:
                    raise AudioSourceError("Image and video_info frame counts differ")
                logging.getLogger(__name__).info(
                    "[SHARP VR Video] Meta Batch: frames %d-%d / %d%s; final=%s",
                    offset + 1, offset + count, timeline_count,
                    "" if final else " (estimated)", final)
                # Diagnose direct output wiring without changing VHS state.
                for downstream in (prompt or {}).values():
                    inputs = downstream.get("inputs", {})
                    link = inputs.get("images")
                    if (downstream.get("class_type") == "VHS_VideoCombine"
                            and isinstance(link, (list, tuple)) and len(link) == 2
                            and str(link[0]) == str(unique_id) and link[1] == 0):
                        manager_link = inputs.get("meta_batch")
                        if (not isinstance(manager_link, (list, tuple)) or len(manager_link) != 2
                                or str(manager_link[0]) != str((prompt or {}).get(str(unique_id), {}).get("inputs", {}).get("meta_batch", [None])[0]) or manager_link[1] != 0):
                            logging.warning("[SHARP VR Video] Video Combine must use the same Meta Batch Manager")
                        elif offset > 0 and not meta_batch.outputs:
                            logging.warning("[SHARP VR Video] No existing VHS encoder at a later chunk; check batch wiring and queue settings")
            except Exception as exc:
                # Do not restart animation on every chunk: use fixed target instead.
                logging.warning("[SHARP VR Video] Meta Batch timeline unavailable (%s); using fixed camera targets", str(exc) if isinstance(exc, AudioSourceError) else type(exc).__name__)
                animated = {}
                timeline_valid = False
        out = None
        totals = dict(reconstruct=0.0, export=0.0, browser=0.0, decode=0.0, total=0.0)
        reconstruction_stages = dict(input_model=0.0, encode=0.0, decode=0.0, unproject=0.0)
        reconstruction_samples = 0
        geometry_stages = dict(unproject_prepare=0.0, unproject_svd=0.0, unproject_post=0.0)
        geometry_samples = 0
        browser_stages = dict(load=0.0, ready=0.0, compose=0.0, pixels=0.0)
        load_samples = 0
        load_fetch = load_parse = load_total = fast_count = direct_count = shared_count = packed_count = 0.0
        browser_samples = 0
        raw_frames = 0
        progress = ProgressBar(count)
        # One browser job in flight; reconstruction stays on ComfyUI's execution
        # thread. Opt out for low-VRAM systems or serial A/B benchmarking.
        overlap = os.environ.get("SHARP_VR_SERIAL", "0") != "1"
        pending = None
        batch_started = time.perf_counter()
        wait_total = 0.0

        def collect(ticket, i):
            nonlocal out, raw_frames, browser_samples, load_samples
            nonlocal load_fetch, load_parse, load_total, fast_count, direct_count, shared_count, packed_count, wait_total
            browser_timings = {}
            waiting = time.perf_counter()
            captured = finish_capture(ticket, browser_timings)
            decoded_started = time.perf_counter()
            wait_total += decoded_started - waiting
            if isinstance(captured, tuple):
                raw, w, h = captured
                if (w, h) != (2 * width, height):
                    raise ValueError(f"Unexpected RGBA size: {w}x{h}")
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[..., :3]
                raw_frames += 1
            else:
                with Image.open(io.BytesIO(captured)) as im:
                    if im.size != (2 * width, height):
                        raise ValueError(f"Expected SBS frame {2 * width}x{height}, got {im.size}")
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            if all(key in browser_timings for key in browser_stages):
                browser_samples += 1
                for key in browser_stages:
                    browser_stages[key] += browser_timings[key]
            if all(key in browser_timings for key in ("fetch", "parse", "fast", "load")):
                load_samples += 1
                load_fetch += browser_timings["fetch"]
                load_parse += browser_timings["parse"]
                load_total += browser_timings["load"]
                fast_count += browser_timings["fast"]
                direct_count += browser_timings.get("direct", 0)
                shared_count += browser_timings.get("shared", 0)
                packed_count += browser_timings.get("packed", 0)
            if out is None:
                out = torch.empty((count, height, 2 * width, 3), dtype=torch.float32, device="cpu")
            if isinstance(captured, tuple):
                # Convert directly into the final CPU IMAGE tensor; avoid a
                # temporary RGB float array and an additional full-frame copy.
                np.divide(arr, np.float32(255.0), out=out[i].numpy(), dtype=np.float32)
            else:
                out[i].copy_(torch.from_numpy(arr))
            totals["browser"] += browser_timings["roundtrip"]
            totals["decode"] += time.perf_counter() - decoded_started
            progress.update(1)
            console_progress.update(1)

        from .display_awake import display_awake
        with display_awake(), tqdm(total=count, desc="SHARP stereo", unit="frame",
                  dynamic_ncols=True, mininterval=0.1) as console_progress:
            try:
                packed_video = (stereo.get('depth_scale') == 1 and stereo.get('splat_scale') == 1 and packed_browser_available())
                frames = video_inference_frames(model, image, focal_length_mm, packed_video=packed_video)
                for i, (ply, ext, intr, timings) in enumerate(frames):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    for key in ("reconstruct", "export"):
                        totals[key] += timings[key]
                    if all(key in timings for key in reconstruction_stages):
                        reconstruction_samples += 1
                        for key in reconstruction_stages:
                            reconstruction_stages[key] += timings[key]
                    if all(key in timings for key in geometry_stages):
                        geometry_samples += 1
                        for key in geometry_stages:
                            geometry_stages[key] += timings[key]
                    try:
                        ext_data = self._validate_matrix(ext, "extrinsics", 4, 4)
                        intr_data = self._validate_matrix(intr, "intrinsics", 3, 3)
                        frame_stereo = stereo.copy() if animated else stereo
                        for key, keys in animated.items():
                            frame_stereo[key] = camera_motion_value(stereo[key], keys, offset + i, timeline_count)
                        # Consume the preceding frame before dispatching another.
                        # The image index and its camera parameters never change.
                        if pending is not None:
                            collect(pending, i - 1)
                            pending = None
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        pending = submit_capture(ply, ext_data, intr_data, node_id=unique_id,
                                                 timeout=int(frame_timeout_seconds), stereo=frame_stereo,
                                                 expected_size=(2 * width, height))
                    finally:
                        del ply
                    if not overlap:
                        collect(pending, i)
                        pending = None
                if pending is not None:
                    collect(pending, count - 1)
                    pending = None
            finally:
                if pending is not None:
                    cancel_capture(pending)
                if 'frames' in locals():
                    frames.close()
                try:
                    clear_video_batch_cache()
                except Exception:
                    logging.getLogger(__name__).warning("[SHARP VR Video] Chunk cache cleanup failed", exc_info=True)
        totals["total"] = time.perf_counter() - batch_started
        logging.getLogger(__name__).info(
            "[SHARP VR Video] Pipeline: %s; exposed browser wait=%.3f s/frame (stage times may overlap)",
            "one-frame lookahead" if overlap else "serial", wait_total / count)
        logging.getLogger(__name__).info(
            "[SHARP VR Video] Performance (%d frames, mean seconds/frame): "
            "reconstruct=%.3f, export+cleanup=%.3f, browser=%.3f, pixels+tensor=%.3f, total=%.3f",
            count, *(totals[key] / count for key in ("reconstruct", "export", "browser", "decode", "total")))
        if reconstruction_samples:
            logging.getLogger(__name__).debug(
                "[SHARP VR Video] Reconstruction detail (%d frames, mean stage seconds): "
                "input+model=%.3f, encode=%.3f, decode=%.3f, unproject=%.3f",
                reconstruction_samples,
                *(reconstruction_stages[key] / reconstruction_samples for key in reconstruction_stages))
        if geometry_samples:
            logging.getLogger(__name__).debug(
                "[SHARP VR Video] Geometry detail (%d frames, mean stage seconds): "
                "prepare=%.3f, solve=%.3f, post=%.3f",
                geometry_samples, *(geometry_stages[key] / geometry_samples for key in geometry_stages))
        if browser_samples:
            logging.getLogger(__name__).debug(
                "[SHARP VR Video] Browser detail (%d profiled frames, raw RGBA=%d/%d, mean seconds/frame): "
                "load=%.3f, ready=%.3f, compose=%.3f, pixels=%.3f",
                browser_samples, raw_frames, count,
                *(browser_stages[key] / browser_samples for key in browser_stages))
        if load_samples:
            logging.getLogger(__name__).debug(
                "[SHARP VR Video] Splat load detail (fast parser=%d/%d, mean seconds/frame): fetch=%.3f, parse+construct=%.3f, setup=%.3f",
                int(fast_count), load_samples, load_fetch / load_samples, load_parse / load_samples,
                max(0.0, load_total - load_fetch - load_parse) / load_samples)
            logging.getLogger(__name__).info(
                "[SHARP VR Video] Browser fast path: direct decoder=%d/%d, shared stereo packing=%d/%d, packed GPU export=%d/%d",
                int(direct_count), load_samples, int(shared_count), load_samples, int(packed_count), load_samples)
        output_audio, fps = resolve_video_audio(video_info, audio, prompt, unique_id, count, fallback_fps,
                                                meta_batch, timeline_count if meta_batch is not None and timeline_valid else None)
        if meta_batch is not None:
            logging.getLogger(__name__).info(
                "[SHARP VR Video] Meta Batch output: %d frames, %.6g fps, audio=%s, final=%s",
                count, fps, "present" if output_audio is not None else "None",
                bool(meta_batch.has_closed_inputs))
        return (out, output_audio, fps)


from .sharp_model import SharpVRVideoLoadModel

NODE_CLASS_MAPPINGS = {
    "SharpVRVideoLoadModel": SharpVRVideoLoadModel,
    "SharpVRVideoPreview": SharpVRVideoPreview,
    "SharpVRVideoBatch": SharpVRVideoBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpVRVideoLoadModel": "SHARP VR Video Model Loader",
    "SharpVRVideoPreview": "SHARP VR Video Preview",
    "SharpVRVideoBatch": "SHARP VR Video Batch",
}
