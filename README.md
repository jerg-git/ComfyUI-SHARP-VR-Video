# ComfyUI-SHARP-VR-Video

Experimental, AI-assisted ComfyUI extension for converting ordinary 2D video frames into full side-by-side stereoscopic frames using Apple SHARP and the gsplat.js renderer. An independent fork of [Well-Made/ComfyUI-SHARP-Gaussian-Stereo](https://github.com/Well-Made/ComfyUI-SHARP-Gaussian-Stereo), focused on batch video and VR experiments. Not affiliated with Apple.

**v0.3.0 — High-resolution stereo rendering update.**

- Native per-eye output resolution: Source (default), 720p, 1080p, 1440p, 2160p (4K), or Custom. Width follows source aspect ratio, rounded to an even integer for presets/custom; SBS width is doubled. Render camera intrinsics are scaled, not the reconstructed scene. This is native re-rendering, not super-resolution.
- Raw RGBA capture up to 128 MiB/frame (including 7680 x 2160 SBS), with exact-length streaming above 32 MiB and PNG fallback. No changes to the global HTTP upload limit.
- The default `render_workers` is **2 (default)** for overlapped stereo rendering. The ComfyUI page starts a secondary embedded worker on demand; no BAT files required. Keep the page visible and open; close old dedicated windows first. **1 (compatibility)** is available for lower resource use. Existing explicit worker selections are preserved via legacy-label migration. Dedicated renderer windows are still supported; if using them, open two for two-worker operation. Check `Worker concurrency` in the console; peak active claims of 2 indicates overlap. Two workers share one GPU, so performance and VRAM use vary. One worker can process all frames if no second worker is available; a disconnected worker with an already-claimed frame may time out.
- Optional **TensorRT for RTX (ONNX FP16)** backend for Windows NVIDIA RTX systems (see below). No mandatory new dependencies, no automatic package installation, no engine or model bundled.

### Measured high-resolution performance

On an RTX 5080, with the same 64-frame 720p source, 32-frame Meta Batches, optional CuPy and TensorRT FP16, the v0.2.14 dual-dedicated-worker test reached 0.313 s/frame at 4K per eye (7680 x 2160 SBS), compared with 1.505 s/frame in the v0.2.12 single-worker PNG capture test: **4.81x throughput**. With one worker and raw capture the intermediate result was 0.489 s/frame. The 1080p and 1440p dual-worker tests reached 0.262 and 0.276 s/frame respectively (from 0.400 and 0.503). All 64 decoded frame RGB checksums and timestamps matched exactly between the one- and two-worker raw 4K runs. These are warmed batch pipeline timings, not total conversion times; they exclude startup, input and final encoding. Results vary. Automatic embedded-worker startup is new in v0.3.0 and needs live browser/GPU verification.

### Optional: TensorRT for RTX (ONNX FP16)

PyTorch remains fully supported, including when TensorRT is absent. The `Auto` backend uses FP16 TensorRT only when a local engine is found in `models/sharp/tensorrt` or the custom `rtx_test_folder` and initializes successfully; otherwise it uses PyTorch. Older workflows without this widget continue to use PyTorch. Explicit TensorRT requires FP16 and fails with an actionable error if unavailable. Auto does not silently change backends after initialization, including across verified VHS Meta Batches. Preview remains PyTorch. Legacy widget labels are accepted and migrated without appearing in the dropdown; the old parity-reference selection now maps to ordinary PyTorch. Normal rendering does not read or write parity reference files.

The backend uses the unofficial [Cool2009/SHARP-onnx](https://huggingface.co/Cool2009/SHARP-onnx) FP16 derivative (revision `7871e64`) of Apple's SHARP checkpoint with TensorRT for RTX `1.7.1.116`. Apple's model license is research-only; review [the model license](LICENSES/APPLE-MODEL-LICENSE.txt) and the derivative repository before using it. Neither the derivative weights nor engine is redistributed here. Use only trusted local engines. TensorRT and PyTorch are not numerically identical; the original PyTorch model is the reference path.

1. If you already have the standalone setup folder, reuse it. Otherwise download the optional `SHARP-VR-Video-TensorRT-RTX-setup.zip` release asset, create `ComfyUI/models/sharp/tensorrt`, extract the kit contents directly into that folder, and close ComfyUI. A custom writable folder outside `custom_nodes` and `python_embeded` is also supported. Run `SETUP_TENSORRT.bat`, review the prompt and provide your ComfyUI Portable `python_embeded/python.exe` path if prompted. It uses an isolated runtime, downloads the pinned FP16 derivative and builds the engine locally. Allow at least 8 GiB free disk; the setup may take several minutes. It is optional and does not modify the normal ComfyUI Python environment.
2. Restart ComfyUI, select FP16 and `Auto` or `TensorRT for RTX (ONNX FP16)` in the Model Loader. Leave `rtx_test_folder` blank to discover the standard location automatically, or enter the custom folder containing `sharp_fp16_rtx_1_7_1_116.engine` and `isolated_python`. Existing compatible setups can be reused without rebuilding. With ComfyUI closed, you may move the entire existing setup folder to the standard location, but relocation itself has not been Windows-tested.
3. If initialization fails, use PyTorch. The setup and runtime are currently tested on Windows Portable with an RTX 5080; other configurations may require adaptation. The optional setup tool uses the previous tested standalone benchmark script, with a generic launcher. The new consolidated integration has not yet been live-tested.

## Features

- IMAGE batch in, full-width SBS IMAGE batch out; compatible with VideoHelperSuite (VHS).
- Editable one-click stereo presets, fixed baseline, parallel off-axis projection, convergence, focal length, depth and splat scale.
- Render-only inward dolly and two-axis orbit controls, independently fixed or animated over the input batch.
- RAM-only intermediate Gaussian transport (PLY or accelerated packed format), frame-based progress and quiet console logs.
- Shared browser renderer independent of the active workflow, plus a dedicated renderer page.
- Retains the interactive single-image preview and optional PLY export.

## What's new in v0.2.10

- Fixed black speckling observed in bright specular reflections under parallel off-axis stereo. The renderer now uses precise float32 depth ordering for parallel stereo and a numerically robust projected-Gaussian eigenvector calculation. Both changes are retained; their individual contribution has not been isolated.
- No highlight compression, image filtering or new repair controls. Normal source RGB is preserved; invalid or out-of-range batch RGB is safely sanitized before 8-bit conversion.
- Windows display-awake protection during active rendering, plus an optional isolated Chrome/Edge renderer launcher (see below). No manual power-plan changes needed.
- New **Launch optimized renderer (Windows)** button on the batch node. No workflow edits, manual BAT launch or queue execution required.
- All v0.2.0 acceleration paths and existing sample workflow retained. No additional dependencies.

On the RTX 5080, a 40-frame 1400 x 784 silent test clip that previously exhibited black speckles rendered cleanly with the two renderer fixes in v0.2.8. Warm 16- and 8-frame batches took 0.440 and 0.461 s/frame respectively. Broader visual and hardware regression testing is still outstanding. If you tried intermediate diagnostic builds, reload the included workflow or remove/re-add the batch node if its obsolete experimental widgets persist.

## What's new in v0.2.0

A major under-the-hood performance update. Existing v0.1.5 workflows, node settings, camera animation, audio/FPS handling and VHS Meta Batch wiring are unchanged. No new mandatory dependencies.

- Overlapping single-frame SHARP inference with browser rendering, with less redundant feature caching, GPU memory cleanup and image conversion.
- Faster binary Gaussian loading, shared stereo geometry preparation, reused WebGL resources, event-driven dispatch and raw RGBA capture.
- Optional CuPy acceleration: a specialized FP32 CUDA covariance solver and a compact GPU-packed transfer format. Both retain automatic reference-path fallbacks when unavailable or rejected by validation.
- Eight-patch encoder batching by default, with memory-error fallback. Full-frame inference remains single-frame; the unsuccessful two-frame experiment was removed.

### Performance on an RTX 5080

| Build / configuration | Warm processing time per frame |
|---|---:|
| v0.1.5 (original baseline, 1400 x 784 input) | 1.850 s |
| v0.1.16 (optional CUDA solver, 1280 x 720 input) | 0.548 s |
| v0.1.17 (browser fast path, same 720p clip) | 0.457 s |
| v0.1.18 (single-frame + packed export, same 720p clip) | 0.430 s |

The final tested configuration processed all 177 frames of a silent 720p clip across 128 + 49 frame Meta Batches. Compared with v0.1.16 on the same clip, the warmed-up final batch was about 27% faster in throughput. The roughly 4.3x comparison with v0.1.5 is illustrative, not a controlled same-resolution benchmark. Results depend on hardware, browser, stereo settings, optional dependencies and input; startup, video loading and encoding overhead are excluded from per-frame timings. The v0.2.0 package consolidates the tested single-frame configuration; broader visual and hardware regression testing remains outstanding. Model weights, inference resolution and Gaussian count are unchanged, but numerical rounding and equivalent covariance eigenbases can produce small rendering differences.

## Known issues

- Some **video-without-audio** workflows can still emit harmless warnings from VideoHelperSuite during caching or Meta Batch processing even when SHARP is functioning correctly. This is an upstream VHS lazy-audio issue; final output can still complete successfully.
- The built-in Queue Manager guard only targets recognized **SHARP VR Video + VHS Meta Batch** continuations under QuietNoise Queue Manager. It does not alter unrelated workflows or other queue implementations.
- Browser and OS background throttling can still reduce rendering throughput if no visible renderer window remains active.
- Variable-frame-rate or resampled clips may produce small timing differences for animated camera moves across Meta Batch segments.

## Installation

1. Clone or extract this repository into `ComfyUI/custom_nodes/ComfyUI-SHARP-VR-Video` (the directory containing `__init__.py`).
2. Install requirements using the Python environment running ComfyUI. For Windows Portable, from its root:

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-SHARP-VR-Video\requirements.txt
   ```

3. Restart ComfyUI and hard-refresh the frontend.
4. The model loader reuses the existing `models/sharp` checkpoint or downloads the official weights (~2.7 GB). No weights are included.

A supported CUDA PyTorch setup and a WebGL-enabled browser are required. Install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) separately for video input/output. FP16 is the default model precision. The shared renderer still needs a browser page open; keep its window visible for best performance.

## Optional: enable faster CUDA processing (NVIDIA GPUs)

**You can skip this entire section.** CuPy is not installed automatically and is not needed for normal operation. Without it, the extension uses PyTorch SVD and the existing PLY transfer path, while retaining the other performance improvements. If CuPy is already working in your ComfyUI environment, you do not need to reinstall it.

### Windows Portable (recommended instructions)

1. Close ComfyUI. Open Command Prompt in your `ComfyUI_windows_portable` folder (the folder containing `python_embeded`). Back up `python_embeded` outside that folder if you want a straightforward rollback.
2. Check the CUDA version used by your ComfyUI PyTorch installation:

   ```bat
   python_embeded\python.exe -c "import torch; print(torch.version.cuda)"
   ```

3. Run **only one** of the following commands, matching the major version printed above:

   **CUDA 12.x** (for example, 12.8):

   ```bat
   python_embeded\python.exe -m pip install "cupy-cuda12x[ctk]"
   ```

   **CUDA 13.x** (for example, 13.0):

   ```bat
   python_embeded\python.exe -m pip install "cupy-cuda13x[ctk]"
   ```

   If the result is `None`, or another unsupported CUDA version, skip the optional acceleration. Do not install both CuPy variants or the source package `cupy` alongside a wheel. For a cautious install, first add `--dry-run` after `pip install` to inspect proposed changes; skip the installation if it would change core dependencies you need to keep. The `[ctk]` extra supplies CUDA components without requiring a system-wide CUDA Toolkit, but a compatible NVIDIA driver is still required. See the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
4. Restart ComfyUI, hard-refresh the browser (Ctrl+F5), and run a short video. No workflow edits or BAT settings are needed.

### Other ComfyUI installations

Use the Python executable/environment that actually launches ComfyUI (activate its virtual environment first, if applicable). Run `python -c "import torch; print(torch.version.cuda)"`, then `python -m pip install "cupy-cuda12x[ctk]"` or `python -m pip install "cupy-cuda13x[ctk]"` as appropriate. Do not install into an unrelated system Python. Existing working CuPy installs can be reused.

### Check that acceleration is working

The console may show `CUDA covariance kernel: Jacobi selected` and `Packed GPU export validated`. Packed transport requires an updated browser worker and neutral depth/splat scales (both 1.0). The first eligible run includes kernel compilation and numerical checks; subsequent frames are faster. If the optional path is missing, fails a recoverable validation/compilation check or is unsuitable, the extension uses its reference path. No need to troubleshoot CuPy just to use the node. Unrecoverable CUDA errors may require restarting ComfyUI.

For troubleshooting, add these optional environment variables before the Python launch command in your Windows BAT file (or export them in your shell before launching):

```bat
set SHARP_VR_COV_KERNEL=0
set SHARP_VR_PACKED=0
```

These disable the optional covariance kernel and packed transfer independently; remove the lines to restore automatic selection. They do not undo pip dependency changes. `SHARP_VR_BROWSER_LEGACY=1` additionally disables the browser fast path. `SHARP_VR_SERIAL=1` disables pipeline overlap for diagnosis. Old `SHARP_VR_FRAME_BATCH` settings are no longer needed or used.

## Quick start

Connect `SHARP VR Video Model Loader` to `SHARP VR Video Batch`. Connect VHS Load Video's IMAGE output to the batch node's image input, then connect `stereo_frames` to VHS Video Combine. Connect the VHS loader `video_info` to the SHARP batch `video_info` input. Connect SHARP `fps` to Video Combine `frame_rate` and SHARP `audio` to Video Combine `audio`. Leave the VHS loader AUDIO output disconnected. Output frames are `[B,H,2W,3]` (left/right SBS).

The [sample workflow](workflow/sharp-vr-video-batch-audio-sample.json) supports Upload and Path via a shared Impact Switch selection. The Path field uses RES4LYF TextBoxConcatenate; install both packs to use this sample. Select your own input video and set the cap to 0 for the full clip. Start with 3–5 frames. For an approximately 1 m source distance and 1 m VR screen distance, try parallel off-axis, a 60–63 mm baseline, 1.0 m convergence, and depth/splat scales of 1.0. Match focal length to the source when possible. These are approximate starting values, not calibrated viewing guarantees.

For single images, connect the same model loader to `SHARP VR Video Preview`.

**Upgrading from v0.1.5:** Replace/update the extension, restart ComfyUI and hard-refresh the browser. Existing workflows and checkpoints can be reused. If migrating from the original upstream or a pre-release fork with different node IDs, replace those old nodes and reconnect their inputs. Keep backups outside `custom_nodes`.

## Limitations

Independent per-frame monocular inference may cause temporal flicker, missing surfaces, and inaccurate scale. This is stereo video, not a 6DoF volumetric video exporter. A browser must remain open; background scheduling is improved but may still be throttled. Long batches can consume significant system RAM for image tensors. Current builds have not been comprehensively tested across hardware and browsers.

## Model download and license

The SHARP checkpoint is downloaded from the official [`apple/Sharp` Hugging Face repository](https://huggingface.co/apple/Sharp). Model weights are not redistributed here.

**Important:** the checkpoint is governed by Apple's [Apple Machine Learning Research Model License Agreement](LICENSES/APPLE-MODEL-LICENSE.txt). The license restricts use to the defined Research Purposes and excludes commercial exploitation, product development, and use in a commercial product or service. Read the full agreement before downloading or using the weights.

The SHARP architecture files retain Apple's copyright and license notices. The ComfyUI integration adapted from ComfyUI-Sharp is distributed under GPL-3.0. Third-party components and their licenses are listed in [`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md) and [`UPSTREAM_SOURCES.md`](UPSTREAM_SOURCES.md).

## Acknowledgements and upstream projects

- [Well-Made/ComfyUI-SHARP-Gaussian-Stereo](https://github.com/Well-Made/ComfyUI-SHARP-Gaussian-Stereo) — direct upstream project and original standalone stereo viewer integration. Thank you to the original author for the foundation and encouragement to develop this fork independently.

- [Apple SHARP](https://github.com/apple/ml-sharp) — model architecture, weights, and research.
- [ComfyUI-Sharp](https://github.com/PozzettiAndrea/ComfyUI-Sharp) — ComfyUI integration source adapted for this standalone package.
- [comfy-3d-viewers](https://github.com/PozzettiAndrea/comfy-3d-viewers) — viewer integration source.
- [gsplat.js](https://github.com/dylanebert/gsplat.js) — Gaussian Splat rendering library.

This repository includes the required notices and license texts. It does not claim authorship of the upstream model or third-party viewer components.


## VHS Batch: VR viewing presets

Connect VHS Load Video IMAGE to SHARP VR Video Batch, then its stereo_frames IMAGE output to VHS Video Combine. The output is full-width left/right SBS (B,H,2W,3). Keep any ComfyUI browser page or the dedicated SHARP renderer page open; the workflow and node need not be visible.

| stereo_preset | Baseline | Zero-parallax distance |
|---|---:|---:|
| Legacy / Viewer Controls | Existing relative intensity | Legacy toe-in |
| VR Life-size / Half Body | 60 mm | 2 m |
| VR Life-size / Mixed Shots (default) | 60 mm | 2.5 m |
| VR Life-size / Full Body | 60 mm | 3.5 m |
| VR 3D Cinema / Comfortable | 35 mm | 3 m |
| VR 3D Cinema / Immersive | 50 mm | 3 m |
| Custom / Fixed Stereo | custom_baseline_mm | custom_convergence_m (0 = infinity) |

Selecting a preset fills the actual node widgets once; edit any value afterward. Reload selected preset reapplies the starting values. The backend never derives render settings from the preset name. Manual values are saved in the workflow. Presets set focal length (35 mm, or 30 mm for Legacy), projection, baseline, convergence, relative intensity (30%), and depth/splat scales (1.0). Custom / Fixed Stereo leaves current values unchanged. Focal length should match the original source lens when known. The original custom_baseline_mm and custom_convergence_m names are retained for workflow compatibility and apply in any relevant projection mode. Parallel off-axis uses baseline and convergence, Parallel uses baseline with zero parallax at infinity, Toe-in rotates toward the specified distance (0 = parallel), and Legacy projection modes use the relative intensity rather than fixed baseline/convergence. The batch always outputs SBS. Workflows from the original fork use different node identifiers. Replace its loader, preview and batch nodes with the corresponding SHARP VR Video nodes, reconnect them, and restore your settings before queueing. The embedded viewer reflects queued settings; edit the node widgets, not the iframe controls.

These are experimental starting values, not guarantees of accurate scale or comfort. SHARP estimates geometry independently per frame. Scale errors, temporal flicker and novel-view artifacts can remain. Use a flat SBS VR player and adjust virtual screen distance/size to match the desired zero-parallax plane and source field of view. Test several seconds including close foreground objects and scene cuts before converting an entire video. GPU/browser end-to-end rendering requires local testing.

### Batch RAM-only Gaussian transport

The VHS batch node serializes one frame at a time to binary PLY or, when optional GPU packing and compatible browser settings permit, a compact packed Gaussian stream in RAM. It serves the bytes directly to the browser renderer via a unique, non-cacheable URL. No temporary PLY files are created by the batch node. The job payload is released on completion, failure, timeout or interruption. The single-image preview retains its normal file output behavior. The existing renderer attribute precision is retained; small numerical differences are possible. Browser caching is disabled for these temporary payloads, but operating-system paging and other applications can still cause disk activity. Keep a browser renderer page open.

### Workflow-independent batch renderer

The shared renderer now belongs to the browser document, not to the workflow graph. Switching between workflows, panning, collapsing or removing the node from the visible graph no longer tears down the renderer. A shared WebGL worker is started automatically when the ComfyUI frontend initializes. The node no longer contains its own preview iframe; the single-image preview is unchanged. Use the batch node's **Open dedicated SHARP renderer** button to open an independent browser page if you want to close the ComfyUI UI. The server prefers an active standalone renderer for newly queued frames. Open it before queueing, and leave it running if closing the main UI. Its URL is `http://127.0.0.1:8188/extensions/ComfyUI-SHARP-VR-Video/viewer_gaussian.html?batch=1&worker=1` for a default local installation. Keep at least one renderer page running; do not close the browser during a job. Actual browser background/minimized-tab throttling and GPU suspension are browser-dependent. For best reliability keep the dedicated renderer in a visible browser window. If a browser is closed during a claimed frame, that frame may time out; do not reload or close the currently rendering browser mid-job. The worker performs explicit WebGL draws instead of relying on requestAnimationFrame; it does not continuously render while idle. RAM-only transport and existing stereo parameters are retained.

### Prevent display-idle stalls (Windows, v0.2.1 test)

Some browser/GPU configurations may stall rendering when Windows powers off the display, even if the PC never sleeps. During each active SHARP render batch, the node now requests that Windows keep the display awake, then restores its previous thread execution state on completion, interruption or error. This is automatic on Windows, changes no power-plan settings, needs no admin privileges, and does not prevent manual monitor power-off, locking, or sleep. It is released between Meta Batch chunks and during downstream video encoding; it is not a guarantee against all browser/driver suspension. Disable it if desired with `set SHARP_VR_KEEP_DISPLAY_AWAKE=0` before launching ComfyUI. Non-Windows hosts are unchanged.

For additional protection, start ComfyUI normally, open its UI at `http://127.0.0.1:8188` (or `localhost` with your actual port), then click **Launch optimized renderer (Windows)** on the batch node before queueing. The button is available automatically in existing workflows. It is restricted to direct local Windows ComfyUI connections; remote clients should use the existing **Open dedicated SHARP renderer** button instead. If a standalone renderer is already active, close it while idle and wait a few seconds before launching the optimized one. The button uses the same browser settings and profile as the BAT file, and attempts to avoid duplicate launches via an active-worker check and a short cooldown. Alternatively, double-click `launch_renderer_windows.bat` in this extension folder before queueing. It opens a dedicated renderer in an isolated Chrome or Edge profile with Chromium background timer, renderer priority and window-occlusion throttling switches disabled. Keep that window open and visible. It defaults to `http://127.0.0.1:8188`; for a different local port, pass the server origin as the first argument (e.g. `launch_renderer_windows.bat http://127.0.0.1:8190`). It does not alter your normal browser profile. The separate profile is stored under `%LOCALAPPDATA%\SHARP-VR-Video\RendererProfile` (separate Chrome/Edge subfolders). Close the dedicated window to stop using the switches. Do not close the renderer during processing. Browser switches are best-effort and may change across browser versions.

Keep your usual Windows display timeout; no manual power-setting change is required. To verify automatic protection, optionally shorten the timeout and run a longer job. The monitor should stay on during each SHARP render batch and become eligible for idle power-off afterward. These measures intentionally keep the display on during rendering; monitor-off rendering is not guaranteed.

### Background-tab scheduling

The batch renderer now long-polls the Python server (up to 20 seconds per request), and wakes on Gaussian packing/sorting completion events instead of polling WebGL readiness every 40 ms. A watchdog timer and network-error retry remain. Rendering settings and console progress are retained. A hidden or minimized browser can still be throttled, frozen or GPU-suspended by the browser or OS; for best performance keep a dedicated renderer window visible.

Background optimization v2: normalize the synchronous WASM packing module into a Promise before registering completion callbacks. The packing algorithm is unchanged.

### Batch camera framing (v0.1.x)

The following optional widgets are independent of stereo presets and saved with the workflow. They transform the shared virtual camera before generating the left and right eye poses. They do not alter SHARP inference or the SBS output resolution. New nodes default to the original view. The v0.1.1 widgets retain their original order and new motion widgets are appended, with Fixed as the default. Existing workflows can be opened without intentionally changing their camera offsets; verify settings after upgrading. The interactive single-image preview is unchanged.

| Widget | Default | Meaning |
|---|---:|---|
| `camera_dolly_m` | 0 | Move the camera closer, 0 to 0.4 m, like the preview scroll wheel. |
| `camera_orbit_yaw_deg` | 0 | Orbit horizontally within a constrained ±10° range; positive looks right. |
| `camera_orbit_pitch_deg` | 0 | Orbit vertically within a constrained ±10° range; positive looks up. |

Orbit uses a fixed pivot on the original viewing axis at the convergence distance for fixed stereo modes with finite convergence, otherwise 1 m. Translation is relative to the resulting camera orientation. Fixed convergence remains measured from the final camera; adjust it if necessary after dollying. Reset camera adjustments restores all three neutral values and Fixed motion modes without resetting stereo settings or selected patterns. Fixed mode applies the same offsets to each frame; Animated mode computes new offsets independently for each frame, with no accumulation or bounding-box-driven orbit pivot. Start with 0.02–0.05 m or approximately ±1 degree of orbit; the constrained ranges are intended to keep viewpoint changes modest because larger moves can expose missing geometry and increase flicker. Batch zoom and pan were removed to simplify the controls and avoid unhelpful reframing. Test locally before publishing.

### Animated camera motion (v0.1.x)

Each axis has a **Fixed / Animated** motion selector (Fixed by default). Selecting Animated reveals its pattern selector. Use the existing numeric camera controls as the target values, now in increments of 0.001 m or 0.001 degrees. The dolly range remains 0–0.400 m; each orbit axis remains −10.000 to +10.000 degrees. Camera motion is independent of stereo presets. Fixed mode ignores its pattern.

| Pattern | Dolly | Orbit (both axes) |
|---|---|---|
| 0 → Target | Yes | Yes |
| Target → 0 | Yes | Yes |
| −Target → Target | — | Yes |
| Target → −Target | — | Yes |
| 0 → Target → 0 | Yes | Yes |
| Target → 0 → Target | Yes | Yes |
| −Target → Target → −Target | — | Yes |
| Target → −Target → Target | — | Yes |

The three axes can animate independently at the same time. Each leg automatically uses cosine ease-in/out; reversals slow smoothly to a stop. Orbit targets retain their sign: if the target is −5°, −Target is +5°. The first and last frames use the exact endpoints, a two-leg pattern reverses halfway through the batch (between frames if necessary), and a single-frame input uses the first keyframe. Values between keyframes are calculated at floating-point precision, not rounded to the widget step.

Animations span the supplied IMAGE batch by default. For a continuous move across VHS Meta Batch chunks, connect the same manager to SHARP, the selected VHS loader and Video Combine. The intermediate camera timeline is estimated from source metadata and trimming settings; the final chunk uses the actual final frame count. Variable-rate or resampled clips may have small timing differences. No changes are made to the inference model, baseline, convergence, output resolution, shared renderer or RAM-only transport. Use small offsets to limit missing-geometry artifacts and test before committing to a long clip.


### Optional audio and FPS (v0.1.x)

The batch node accepts optional `video_info` (VHS_VIDEOINFO) and `audio` (AUDIO) inputs and returns `stereo_frames`, `audio`, and `fps`. Connect VHS Load Video (Upload or local Path; including FFmpeg variants) directly or through Impact Pack Switch (Any) via `video_info` to the batch node, and wire the batch outputs into Video Combine. Do not connect VHS's lazy AUDIO output, which can fail upstream for videos with no audio track. Set Video Combine to a video format supporting audio.

The batch automatically decodes the first audio stream into 48 kHz stereo float PCM once, using FFmpeg already available to VHS or on PATH, and matches the loaded FPS and selected time window. No audio stream returns `None`; VHS Video Combine can then export video only. Explicit standard AUDIO takes priority. Without video_info, the fallback FPS (default 24) is used; image processing is unchanged. Audio errors are logged and do not abort rendering. The sample TextBoxConcatenate path field is supported when it contains literal text with no external input or separator. Other linked filename/timing widgets, URL sources, mismatched frame counts, and other upstream loader types are not supported for automatic extraction; provide explicit audio instead. Timing assumes the image batch retains the same frame order and temporal sampling as the supplied video_info. Do not route the problematic VHS AUDIO output to the explicit audio input.

Semantic selectors use ComfyUI's explicit COMBO format. Actual appearance depends on the frontend version and UI mode.

### Meta Batch and source switching

Connect the same VHS Meta Batch Manager to the active VHS loader, SHARP `meta_batch`, and Video Combine. Use VHS loader format None or AnimateDiff, an unmodified frame sequence, and a video output format. Start at 32 frames per batch. Audio is returned only on the final chunk, covering the entire processed clip; no-audio clips return None. When using Impact Switch (Any), use the selected-value output, select_on_prompt, and identical selection on IMAGE and video_info switches. Literal selections, ImpactInt and supported integer primitives work; arbitrary runtime-computed choices are not resolved. Do not connect VHS AUDIO.

The Cross-eyed / Freeview preset sets Legacy / Toe-in and stereo intensity -45%, with depth and splat scales 1.0. No extra eye-order widget.
