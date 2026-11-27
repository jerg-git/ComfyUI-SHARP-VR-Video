"""Standalone SHARP model download/configuration and ComfyUI model cache.

Adapted from ComfyUI-Sharp (GPL-3.0), nodes/load_model.py.
The Apple SHARP model implementation is vendored separately under sharp/.
"""

import logging
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

log = logging.getLogger("sharp_vr_video")


def _log(message: str) -> None:
    print(f"[SHARP VR Video] {message}", file=sys.stderr, flush=True)


try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("sharp", MODELS_DIR)
    except Exception:
        pass
except ImportError:
    folder_paths = None
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_model_patcher = None
_model_config = None


def _find_local_checkpoint():
    if folder_paths is not None:
        try:
            hit = folder_paths.get_full_path("sharp", SHARP_FILENAME)
            if hit and os.path.isfile(hit):
                return hit
        except Exception:
            pass
    direct = os.path.join(MODELS_DIR, SHARP_FILENAME)
    return direct if os.path.isfile(direct) else None


def _hf_token():
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _comfy_tqdm():
    try:
        import comfy.utils
        import tqdm as tqdm_module
    except ImportError:
        return None
    state = {"bar": None, "total": 0, "done": 0}

    class ComfyProgress(tqdm_module.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.total and self.total > 0 and state["bar"] is None:
                state["total"] = self.total
                state["bar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            result = super().update(n)
            if n and state["bar"] and state["total"]:
                state["done"] = min(state["done"] + n, state["total"])
                state["bar"].update_absolute(state["done"], state["total"])
            return result

    return ComfyProgress


def _download_with_retry(**kwargs):
    token = _hf_token()
    if token:
        kwargs.setdefault("token", token)
    delays = (5, 15, 30, 60)
    for attempt in range(len(delays) + 1):
        try:
            return hf_hub_download(**kwargs)
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt == len(delays):
                raise
            delay = delays[attempt]
            _log(f"Hugging Face HTTP {status}; retry {attempt + 1}/5 in {delay}s")
            time.sleep(delay)


def resolve_precision(precision="fp16"):
    import comfy.model_management
    device = comfy.model_management.get_torch_device()
    if precision == "auto":
        if comfy.model_management.should_use_bf16(device):
            dtype = torch.bfloat16
        elif comfy.model_management.should_use_fp16(device):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        if precision not in _DTYPE_MAP:
            raise ValueError(f"Unsupported SHARP precision: {precision}")
        dtype = _DTYPE_MAP[precision]
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


def ensure_checkpoint():
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = _find_local_checkpoint()
    if path:
        _log(f"Using existing checkpoint: {path} ({os.path.getsize(path) / 1048576:.1f} MB)")
        return path
    _log(f"Downloading {SHARP_FILENAME} from {SHARP_REPO_ID} to {MODELS_DIR}")
    path = _download_with_retry(
        repo_id=SHARP_REPO_ID,
        filename=SHARP_FILENAME,
        local_dir=MODELS_DIR,
        tqdm_class=_comfy_tqdm(),
    )
    _log(f"Checkpoint ready: {path} ({os.path.getsize(path) / 1048576:.1f} MB)")
    return path


def _load_sharp_model(config):
    """Create/cache the original ComfyUI ModelPatcher-backed SHARP predictor."""
    global _model_patcher, _model_config
    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    from .sharp import PredictorParams, create_predictor

    config = dict(config)
    if _model_patcher is None or _model_config != config:
        dtype = _DTYPE_MAP[config["dtype"]]
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)

        _log(f"Loading checkpoint: {config['model_path']} ({config['dtype']})")
        state_dict = comfy.utils.load_torch_file(config["model_path"])

        # The official SHARP checkpoint stores its floating-point tensors in
        # FP32. With assign=True, load_state_dict replaces the meta-created
        # parameters with checkpoint tensors and therefore preserves FP32.
        # Convert the checkpoint tensors explicitly to the selected model
        # precision before assigning them to the architecture.
        source_dtypes = {}
        converted_tensors = 0
        for key, tensor in state_dict.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                source_key = str(tensor.dtype)
                source_dtypes[source_key] = source_dtypes.get(source_key, 0) + 1
                if tensor.dtype != dtype:
                    state_dict[key] = tensor.to(dtype=dtype)
                    converted_tensors += 1

        _log(
            f"Checkpoint floating tensor dtypes: {source_dtypes}; "
            f"converted to {config['dtype']}: {converted_tensors}"
        )
        _log("Initializing SHARP architecture on meta device")
        with torch.device("meta"):
            predictor = create_predictor(
                PredictorParams(), dtype=dtype, device=None, operations=operations
            )
        predictor.load_state_dict(state_dict, strict=False, assign=True)

        for name, buffer in list(predictor.named_buffers()):
            if buffer.device.type == "meta":
                parent = predictor
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent._buffers[parts[-1]] = torch.zeros_like(buffer, device="cpu")
        predictor.eval()
        if comfy.model_management.force_channels_last():
            predictor.to(memory_format=torch.channels_last)
        comfy.model_management.archive_model_dtypes(predictor)
        parameter_dtype = next(
            (parameter.dtype for parameter in predictor.parameters()
             if parameter.is_floating_point()),
            None,
        )
        if parameter_dtype != dtype:
            raise RuntimeError(
                "SHARP parameter dtype mismatch after checkpoint loading: "
                f"expected {dtype}, got {parameter_dtype}"
            )

        _model_patcher = comfy.model_patcher.ModelPatcher(
            predictor, load_device=load_device, offload_device=offload_device
        )
        _model_config = config
        _log(
            f"SHARP model initialized ({config['dtype']}; "
            f"verified parameter dtype={parameter_dtype})"
        )
    return _model_patcher


class SharpVRVideoLoadModel:
    """Download/locate SHARP weights and emit a lightweight model config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": ("COMBO", {"options": ["auto", "bf16", "fp16", "fp32"], 
                    "default": "fp16",
                    "tooltip": "FP16 is the default. You can also select auto, BF16, or FP32.",
                }),
            }
        }

    RETURN_TYPES = ("SHARP_VR_VIDEO_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "load_model"
    CATEGORY = "SHARP"

    def load_model(self, precision="fp16"):
        dtype_name = resolve_precision(precision)
        model_path = ensure_checkpoint()
        config = {"model_path": model_path, "precision": precision, "dtype": dtype_name}
        _log(f"Model config ready: precision={precision} -> {dtype_name}")
        return (config,)
