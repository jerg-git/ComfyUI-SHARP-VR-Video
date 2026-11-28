# SHARP VR Video v0.3.0 — Faster high-resolution stereo video

Major high-resolution rendering update, building on the v0.2.x performance improvements.

## What's new

- **Up to 4.8x faster 4K-per-eye rendering pipeline** in development benchmarks: lossless raw RGBA transfer replaces the expensive PNG path for frames up to 128 MiB. No global HTTP upload limit changes.
- **Dual render workers**: `2 (default)` overlaps stereo frame jobs while keeping neural inference sequential and output frame order intact. A second embedded browser worker starts on demand. `1 (compatibility)` remains available; the existing dedicated renderer launchers are optional.
- **Per-eye render resolution**: Source, 720p, 1080p, 1440p, 2160p (4K), or Custom. Preserves aspect ratio and camera framing; native Gaussian re-rendering, not an AI upscaler.
- **Optional TensorRT for RTX (ONNX FP16)**: use a trusted, locally built engine on supported Windows NVIDIA RTX systems. Auto discovers an FP16-compatible local engine in `models/sharp/tensorrt` or a custom folder and falls back to PyTorch if initial engine setup fails. It never switches silently mid-video. Explicit TensorRT mode reports errors. Preview remains PyTorch.
- Retains parallel-projection fixes, optional CuPy acceleration, camera animation, audio/FPS, VHS Meta Batch support and Queue Manager protection. Existing workflow and mandatory dependencies unchanged.

## Development benchmark (RTX 5080)

| Output per eye | Earlier v0.2.12 test, 1 worker | v0.2.14 test, 2 workers | Throughput improvement |
|---|---:|---:|---:|
| 1080p | 0.400 s/frame | 0.262 s/frame | 1.53x |
| 1440p | 0.503 s/frame | 0.276 s/frame | 1.82x |
| 4K (7680 x 2160 SBS) | 1.505 s/frame | 0.313 s/frame | 4.81x |

Same 720p source clip, 64 frames, Meta Batch 32, warmed second batch, TensorRT FP16 and CuPy, dedicated renderer windows. At 4K, raw capture with one worker reached 0.489 s/frame before enabling the second worker. All 64 decoded RGB frame MD5 checksums, timestamps and durations matched exactly between the one- and two-worker 4K outputs. This verifies output equivalence for that clip/configuration, not general TensorRT-vs-PyTorch equivalence or all hardware. Times exclude startup, video input and final encoding; results vary. The new embedded-worker startup path is not the one used for these benchmark numbers.

## Upgrade

Update the extension, restart ComfyUI and hard-refresh (Ctrl+F5). Keep a ComfyUI browser page open. Two workers are the default for new nodes; select one if needed for compatibility. close any old dedicated windows first, as the server prefers dedicated workers when present. For 4K use smaller VHS Meta Batches (16–32 depending on RAM). No TensorRT installation is required; ordinary PyTorch workflows still work. Optional TensorRT setup instructions are in the README; extract the separate kit into `ComfyUI/models/sharp/tensorrt` for automatic discovery. Apple's model license limits use to non-commercial scientific research and academic development.

The consolidated build passed local bridge, ordering, cancellation, timeout, audio/timeline and mocked Auto-selection checks. Actual integrated embedded-WebGL startup, NVIDIA engine initialization and wider hardware compatibility require live testing. Silent clips can still trigger upstream VHS lazy-audio cache warnings.

# v0.2.10 — Parallel stereo rendering fix and renderer launcher

- Added a **Launch optimized renderer (Windows)** button on the batch node to open the optional isolated browser renderer without leaving the workflow. Direct local Windows connections only; manual BAT and ordinary dedicated renderer options remain available.
- Fixed black speckling around bright specular highlights observed with parallel off-axis stereo, using precise float32 depth sorting and more robust Gaussian projection shader math.
- Added Windows display-awake protection during active rendering and an optional isolated Chrome/Edge renderer launcher to reduce background throttling. Normal Windows power settings remain unchanged.
- Safe batch RGB clipping before 8-bit conversion. No highlight compression or cosmetic repair is applied.
- Retains the v0.2.0 speedups, optional CuPy acceleration, existing camera controls, audio/FPS handling, VHS Meta Batch compatibility and Queue Manager protection.
- No new mandatory dependencies or changes to the included workflow.

The renderer fix was visually verified in v0.2.8 on an RTX 5080: all 40 frames of the affected 1400 x 784 clip completed, with warm processing times of 0.440–0.461 seconds/frame. The two fixes were tested together, so their individual contributions are not isolated. Broader compatibility remains to be tested.

Update the extension, restart ComfyUI and hard-refresh the browser (Ctrl+F5). The dedicated Windows renderer launcher is optional. Existing CuPy installations do not need to be changed. Intermediate diagnostic widgets have been removed; if upgrading from those test builds, reload the included workflow or replace the batch node if obsolete widgets persist.

# SHARP VR Video v0.2.0 — Major performance update

A substantial under-the-hood speed upgrade to batch stereoscopic video generation, with **no workflow changes required** for v0.1.5 users.

## What's improved

- Overlapped SHARP reconstruction and browser rendering, with reduced caching, memory cleanup and conversion overhead.
- Faster Gaussian decoding, shared stereo geometry preparation, reused WebGL resources and direct RGBA output.
- Optional CUDA acceleration (CuPy): specialized FP32 covariance decomposition and compact GPU-packed Gaussian transfer. Automatic fallback to PyTorch SVD / PLY when the optional paths are unavailable or rejected.
- Single-frame neural inference retained for responsiveness and stability; eight-patch encoder batching by default.
- Existing audio/FPS passthrough, animated camera controls, VHS Meta Batch compatibility and Queue Manager protection preserved.

## Performance

On the RTX 5080 development machine, the final tested single-frame configuration reached **0.430 seconds/frame** in the warmed-up final 49-frame batch of a 177-frame 720p video. The previous v0.1.16 build took 0.548 seconds/frame on the same clip. Earlier v0.1.5 tests measured 1.850 seconds/frame at a different source resolution, so the approximately 4.3x improvement is illustrative, not a controlled apples-to-apples benchmark. Actual gains vary; timings exclude video I/O and encoding. The optional CuPy paths were enabled in the final benchmark.

## Upgrading

Update the extension, restart ComfyUI and hard-refresh the browser (Ctrl+F5). No new mandatory dependencies, node settings or sample workflow changes. For optional CUDA acceleration, follow the simple CUDA 12/13 installation instructions in the README. Users without CuPy can continue to use the normal SVD/PLY paths.

The model weights, inference resolution and Gaussian count are unchanged. Small numerical/rendering differences are possible. The consolidated package still needs final visual and compatibility regression testing. Silent videos may produce an upstream VHS cache/audio warning even when rendering completes. Apple's SHARP model research-only license remains unchanged.
