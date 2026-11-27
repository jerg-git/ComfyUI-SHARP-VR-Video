# ComfyUI-SHARP-VR-Video

Experimental, AI-assisted ComfyUI extension for converting ordinary 2D video frames into full side-by-side stereoscopic frames using Apple SHARP and the gsplat.js renderer. An independent fork of [Well-Made/ComfyUI-SHARP-Gaussian-Stereo](https://github.com/Well-Made/ComfyUI-SHARP-Gaussian-Stereo), focused on batch video and VR experiments. Not affiliated with Apple.

**v0.1.2 release candidate.** Local end-to-end GPU testing is required before publishing. SHARP model use is subject to Apple's research-only license (see below).

## Features

- IMAGE batch in, full-width SBS IMAGE batch out; compatible with VideoHelperSuite (VHS).
- Editable one-click stereo presets, fixed baseline, parallel off-axis projection, convergence, focal length, depth and splat scale.
- Render-only inward dolly and two-axis orbit controls, independently fixed or animated over the input batch.
- RAM-only intermediate PLY transport, frame-based progress and quiet console logs.
- Shared browser renderer independent of the active workflow, plus a dedicated renderer page.
- Retains the interactive single-image preview and optional PLY export.

## Installation

1. Clone or extract this repository into `ComfyUI/custom_nodes/ComfyUI-SHARP-VR-Video` (the directory containing `__init__.py`).
2. Install requirements using the Python environment running ComfyUI. For Windows Portable, from its root:

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-SHARP-VR-Video\requirements.txt
   ```

3. Restart ComfyUI and hard-refresh the frontend.
4. The model loader reuses the existing `models/sharp` checkpoint or downloads the official weights (~2.7 GB). No weights are included.

A supported CUDA PyTorch setup and a WebGL-enabled browser are required. Install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) separately for video input/output. FP16 is the default model precision. The shared renderer still needs a browser page open; keep its window visible for best performance.

## Quick start

Connect `SHARP VR Video Model Loader` to `SHARP VR Video Batch`. Connect VHS Load Video's IMAGE output to the batch node's image input, then connect `stereo_frames` to VHS Video Combine. Connect the VHS loader `video_info` to the SHARP batch `video_info` input. Connect SHARP `fps` to Video Combine `frame_rate` and SHARP `audio` to Video Combine `audio`. Leave the VHS loader AUDIO output disconnected. Output frames are `[B,H,2W,3]` (left/right SBS).

A new [batch audio example](workflow/sharp-vr-video-batch-audio.json) starts with a five-frame test cap; select your own input video and set the cap to 0 for the full clip. Start with 3–5 frames. For an approximately 1 m source distance and 1 m VR screen distance, try parallel off-axis, a 60–63 mm baseline, 1.0 m convergence, and depth/splat scales of 1.0. Match focal length to the source when possible. These are approximate starting values, not calibrated viewing guarantees.

For single images, connect the same model loader to `SHARP VR Video Preview`; see the included [preview example](workflow/gaussian-stereo-workflow.json). The legacy preview screenshot is inherited from the original project.

**Migrating from previous ZIP builds:** Node IDs and model socket type have changed so the upstream extension can be installed separately. Replace old SHARP nodes in saved workflows, reconnect inputs, and copy settings. Remove the earlier experimental fork from `custom_nodes` (back it up outside that directory first). The model checkpoint can be reused.

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

### Batch RAM-only PLY transport

The VHS batch node now serializes one frame at a time to binary PLY bytes in RAM and serves those bytes directly to the existing browser renderer via a unique, non-cacheable URL. No temporary PLY files are created by the batch node. The job payload is released on completion, failure, timeout or interruption. The single-image preview retains its normal file output behavior. Rendering settings and Gaussian precision are unchanged. Browser caching is disabled for these temporary payloads, but operating-system paging and other applications can still cause disk activity. Keep a browser renderer page open.

### Workflow-independent batch renderer

The shared renderer now belongs to the browser document, not to the workflow graph. Switching between workflows, panning, collapsing or removing the node from the visible graph no longer tears down the renderer. A shared WebGL worker is started automatically when the ComfyUI frontend initializes. The node no longer contains its own preview iframe; the single-image preview is unchanged. Use the batch node's **Open dedicated SHARP renderer** button to open an independent browser page if you want to close the ComfyUI UI. The server prefers an active standalone renderer for newly queued frames. Open it before queueing, and leave it running if closing the main UI. Its URL is `http://127.0.0.1:8188/extensions/ComfyUI-SHARP-VR-Video/viewer_gaussian.html?batch=1&worker=1` for a default local installation. Keep at least one renderer page running; do not close the browser during a job. Actual browser background/minimized-tab throttling and GPU suspension are browser-dependent. For best reliability keep the dedicated renderer in a visible browser window. If a browser is closed during a claimed frame, that frame may time out; do not reload or close the currently rendering browser mid-job. The worker performs explicit WebGL draws instead of relying on requestAnimationFrame; it does not continuously render while idle. RAM-only PLY transport and existing stereo parameters are unchanged.

### Background-tab scheduling

The batch renderer now long-polls the Python server (up to 20 seconds per request), and wakes on Gaussian packing/sorting completion events instead of polling WebGL readiness every 40 ms. A watchdog timer and network-error retry remain. Rendering settings, PLY transport and console progress are unchanged. A hidden or minimized browser can still be throttled, frozen or GPU-suspended by the browser or OS; for best performance keep a dedicated renderer window visible.

Background optimization v2: normalize the synchronous WASM packing module into a Promise before registering completion callbacks. The packing algorithm is unchanged.

### Batch camera framing (v0.1.2)

The following optional widgets are independent of stereo presets and saved with the workflow. They transform the shared virtual camera before generating the left and right eye poses. They do not alter SHARP inference or the SBS output resolution. New nodes default to the original view. The v0.1.1 widgets retain their original order and new motion widgets are appended, with Fixed as the default. Existing workflows can be opened without intentionally changing their camera offsets; verify settings after upgrading. The interactive single-image preview is unchanged.

| Widget | Default | Meaning |
|---|---:|---|
| `camera_dolly_m` | 0 | Move the camera closer, 0 to 0.4 m, like the preview scroll wheel. |
| `camera_orbit_yaw_deg` | 0 | Orbit horizontally within a constrained ±10° range; positive looks right. |
| `camera_orbit_pitch_deg` | 0 | Orbit vertically within a constrained ±10° range; positive looks up. |

Orbit uses a fixed pivot on the original viewing axis at the convergence distance for fixed stereo modes with finite convergence, otherwise 1 m. Translation is relative to the resulting camera orientation. Fixed convergence remains measured from the final camera; adjust it if necessary after dollying. Reset camera adjustments restores all three neutral values and Fixed motion modes without resetting stereo settings or selected patterns. Fixed mode applies the same offsets to each frame; Animated mode computes new offsets independently for each frame, with no accumulation or bounding-box-driven orbit pivot. Start with 0.02–0.05 m or approximately ±1 degree of orbit; the constrained ranges are intended to keep viewpoint changes modest because larger moves can expose missing geometry and increase flicker. Batch zoom and pan were removed to simplify the controls and avoid unhelpful reframing. Test locally before publishing.

### Animated camera motion (v0.1.2)

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

Animations span the supplied IMAGE batch, not an externally timed video. If VHS or a workflow splits a clip into multiple batches, the animation restarts on each batch. Use a single batch for one continuous move; two-frame batches cannot depict an intermediate reversal. No changes are made to the inference model, baseline, convergence, output resolution, shared renderer or RAM-only transport. Use small offsets to limit missing-geometry artifacts and test before committing to a long clip.


### Optional audio and FPS (v0.1.2)

The batch node accepts optional `video_info` (VHS_VIDEOINFO) and `audio` (AUDIO) inputs and returns `stereo_frames`, `audio`, and `fps`. Connect VHS Load Video (Upload or local Path; including FFmpeg variants) directly via `video_info` to the batch node, and wire the batch outputs into Video Combine. Do not connect VHS's lazy AUDIO output, which can fail upstream for videos with no audio track. Set Video Combine to a video format supporting audio.

The batch automatically decodes the first audio stream into 48 kHz stereo float PCM once, using FFmpeg already available to VHS or on PATH, and matches the loaded FPS and selected time window. No audio stream returns `None`; VHS Video Combine can then export video only. Explicit standard AUDIO takes priority. Without video_info, the fallback FPS (default 24) is used; image processing is unchanged. Audio errors are logged and do not abort rendering. Linked filename/timing widgets, URL sources, VHS Meta Batch, mismatched frame counts, and other upstream loader types are not supported for automatic extraction; provide explicit audio instead. Timing assumes the image batch retains the same frame order and temporal sampling as the supplied video_info. Do not route the problematic VHS AUDIO output to the explicit audio input.

Semantic selectors use ComfyUI's explicit COMBO format. Actual appearance depends on the frontend version and UI mode.
