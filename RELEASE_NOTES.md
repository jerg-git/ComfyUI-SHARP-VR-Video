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
