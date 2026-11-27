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


class SharpVRVideoBatch(GaussianStereoPreview):
    """Run SHARP per input frame and capture the original gsplat viewer's stereo canvas."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SHARP_VR_VIDEO_MODEL_CONFIG",), "image": ("IMAGE",),
            "focal_length_mm": ("FLOAT", {"default":35.0,"min":0.0,"max":500.0,"step":0.1}),
            "frame_timeout_seconds": ("INT", {"default":180,"min":30,"max":900,"step":10})},
            "optional": {
                "stereo_preset": (STEREO_PRESETS, {"default": "VR Life-size / Mixed Shots",
                    "tooltip": "Selecting a preset fills the editable widgets once. The actual widget values are used for rendering."}),
                "custom_baseline_mm": ("FLOAT", {"default":60.0,"min":0.0,"max":100.0,"step":0.1,
                    "tooltip": "Virtual camera separation in millimeters, used in fixed projection modes."}),
                "custom_convergence_m": ("FLOAT", {"default":2.5,"min":0.0,"max":100.0,"step":0.1,
                    "tooltip": "Zero-parallax distance for off-axis or toe-in projection. 0 means infinity."}),
                "projection_mode": (PROJECTION_MODES, {"default": "Parallel off-axis"}),
                "stereo_intensity": ("FLOAT", {"default":30.0,"min":-100.0,"max":100.0,"step":1.0,
                    "tooltip": "Relative stereo intensity (percent). Only used in the Legacy projection modes."}),
                "depth_scale": ("FLOAT", {"default":1.0,"min":0.1,"max":3.0,"step":0.01}),
                "splat_size_scale": ("FLOAT", {"default":1.0,"min":0.01,"max":100.0,"step":0.01}),
                "camera_dolly_m": ("FLOAT", {"default":0.0,"min":0.0,"max":0.4,"step":0.01,
                    "tooltip": "Move both cameras closer in metres (like the preview wheel). 0 = original camera position."}),
                "camera_orbit_yaw_deg": ("FLOAT", {"default":0.0,"min":-10.0,"max":10.0,"step":0.5,
                    "tooltip": "Orbit horizontally about the fixed convergence distance (1 m in legacy/infinite modes). Positive looks right."}),
                "camera_orbit_pitch_deg": ("FLOAT", {"default":0.0,"min":-10.0,"max":10.0,"step":0.5,
                    "tooltip": "Orbit vertically about the same fixed pivot. Positive looks up."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = False
    RETURN_NAMES = ("stereo_frames",)
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
                  camera_orbit_pitch_deg=0.0):
        import io
        import numpy as np
        import torch
        import comfy.model_management
        from comfy.utils import ProgressBar
        from PIL import Image
        from tqdm import tqdm
        from .sharp_inference import run_sharp_inference
        from .batch_bridge import capture

        if getattr(image, "ndim", None) != 4 or image.shape[0] < 1 or image.shape[-1] != 3:
            raise ValueError("Expected RGB IMAGE batch [B,H,W,3]")
        if unique_id is None:
            raise ValueError("Batch node requires its ComfyUI unique_id")
        stereo = resolve_stereo_settings(projection_mode, custom_baseline_mm, custom_convergence_m,
                                         stereo_intensity, depth_scale, splat_size_scale,
                                         camera_dolly_m, camera_orbit_yaw_deg, camera_orbit_pitch_deg)
        count, height, width, _ = image.shape
        out = None
        progress = ProgressBar(count)
        with tqdm(total=count, desc="SHARP stereo", unit="frame",
                  dynamic_ncols=True, mininterval=0.1) as console_progress:
            for i in range(count):
                comfy.model_management.throw_exception_if_processing_interrupted()
                ply, ext, intr = run_sharp_inference(model, image[i:i+1], None, focal_length_mm, quiet=True)
                ext_data = self._validate_matrix(ext, "extrinsics", 4, 4)
                intr_data = self._validate_matrix(intr, "intrinsics", 3, 3)
                try:
                    png = capture(ply, ext_data, intr_data, node_id=unique_id,
                                  timeout=int(frame_timeout_seconds), stereo=stereo)
                finally:
                    # Release this frame before running inference on the next one.
                    del ply
                with Image.open(io.BytesIO(png)) as im:
                    if im.size != (2 * width, height):
                        raise ValueError(f"Expected SBS frame {2 * width}x{height}, got {im.size}")
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                if out is None:
                    out = torch.empty((count, height, 2 * width, 3), dtype=torch.float32, device="cpu")
                out[i].copy_(torch.from_numpy(arr))
                progress.update(1)
                console_progress.update(1)
        return (out,)


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
