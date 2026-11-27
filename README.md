# ComfyUI-SHARP-VR-Video

Experimental, AI-assisted ComfyUI extension for converting ordinary 2D video frames into full side-by-side stereoscopic frames using Apple SHARP and the gsplat.js renderer. An independent fork of [Well-Made/ComfyUI-SHARP-Gaussian-Stereo](https://github.com/Well-Made/ComfyUI-SHARP-Gaussian-Stereo), focused on batch video and VR experiments. Not affiliated with Apple.

**v0.1.1 release candidate.** Local end-to-end GPU testing is required before publishing. SHARP model use is subject to Apple's research-only license (see below).

## Features

- IMAGE batch in, full-width SBS IMAGE batch out; compatible with VideoHelperSuite (VHS).
- Editable one-click stereo presets, fixed baseline, parallel off-axis projection, convergence, focal length, depth and splat scale.
- Render-only inward dolly and two-axis orbit controls for the entire batch.
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

Connect `SHARP VR Video Model Loader` to `SHARP VR Video Batch`. Connect VHS Load Video's IMAGE output to the batch node's image input, then connect `stereo_frames` to VHS Video Combine. Match the source frame rate and optionally connect source audio directly to the video combiner. Output frames are `[B,H,2W,3]` (left/right SBS).

Start with 3–5 frames. For an approximately 1 m source distance and 1 m VR screen distance, try parallel off-axis, a 60–63 mm baseline, 1.0 m convergence, and depth/splat scales of 1.0. Match focal length to the source when possible. These are approximate starting values, not calibrated viewing guarantees.

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

Selecting a preset fills the actual node widgets once; edit any value afterward. Reload selected preset reapplies the starting values. The backend never derives render settings from the preset name. Manual values are saved in the workflow. Presets set focal length (35 mm, or 30 mm for Legacy), projection, baseline, convergence, relative intensity (30%), and depth/splat scales (1.0). Custom / Fixed Stereo leaves current values unchanged. Focal length should match the original source lens when known. The original custom_baseline_mm and custom_convergence_m names are retained for workflow compatibility and apply in any relevant projection mode. Parallel off-axis uses baseline and convergence, Parallel uses baseline with zero parallax at infinity, Toe-in rotates toward the specified distance (0 = parallel), and Legacy projection modes use the relative intensity rather than fixed baseline/convergence. The batch always outputs SBS. Workflows from the original fork use different node identifiers. Replace its loader, preview and batch nodes with the corresponding SHARP VR Video nodes, reconnect them, and restore your settings before queueing. The embedded viewer reflects queued settings; edit the node widgets, not the iframe controls.

These are experimental starting values, not guarantees of accurate scale or comfort. SHARP estimates geometry independently per frame. Scale errors, temporal flicker and novel-view artifacts can remain. Use a flat SBS VR player and adjust virtual screen distance/size to match the desired zero-parallax plane and source field of view. Test several seconds including close foreground objects and scene cuts before converting an entire video. GPU/browser end-to-end rendering requires local testing.

### Batch RAM-only PLY transport

The VHS batch node now serializes one frame at a time to binary PLY bytes in RAM and serves those bytes directly to the existing browser renderer via a unique, non-cacheable URL. No temporary PLY files are created by the batch node. The job payload is released on completion, failure, timeout or interruption. The single-image preview retains its normal file output behavior. Rendering settings and Gaussian precision are unchanged. Browser caching is disabled for these temporary payloads, but operating-system paging and other applications can still cause disk activity. Keep a browser renderer page open.

### Workflow-independent batch renderer

The shared renderer now belongs to the browser document, not to the workflow graph. Switching between workflows, panning, collapsing or removing the node from the visible graph no longer tears down the renderer. A shared WebGL worker is started automatically when the ComfyUI frontend initializes. The node no longer contains its own preview iframe; the single-image preview is unchanged. Use the batch node's **Open dedicated SHARP renderer** button to open an independent browser page if you want to close the ComfyUI UI. The server prefers an active standalone renderer for newly queued frames. Open it before queueing, and leave it running if closing the main UI. Its URL is `http://127.0.0.1:8188/extensions/ComfyUI-SHARP-VR-Video/viewer_gaussian.html?batch=1&worker=1` for a default local installation. Keep at least one renderer page running; do not close the browser during a job. Actual browser background/minimized-tab throttling and GPU suspension are browser-dependent. For best reliability keep the dedicated renderer in a visible browser window. If a browser is closed during a claimed frame, that frame may time out; do not reload or close the currently rendering browser mid-job. The worker performs explicit WebGL draws instead of relying on requestAnimationFrame; it does not continuously render while idle. RAM-only PLY transport and existing stereo parameters are unchanged.

### Background-tab scheduling

The batch renderer now long-polls the Python server (up to 20 seconds per request), and wakes on Gaussian packing/sorting completion events instead of polling WebGL readiness every 40 ms. A watchdog timer and network-error retry remain. Rendering settings, PLY transport and console progress are unchanged. A hidden or minimized browser can still be throttled, frozen or GPU-suspended by the browser or OS; for best performance keep a dedicated renderer window visible.

Background optimization v2: normalize the synchronous WASM packing module into a Promise before registering completion callbacks. The packing algorithm is unchanged.

### Batch camera framing (v0.1.1)

The following optional widgets are independent of stereo presets and saved with the workflow. They transform the shared virtual camera before generating the left and right eye poses. They do not alter SHARP inference or the SBS output resolution. New nodes default to the original view. When upgrading from v0.1.1, recreate the batch node and restore your settings: the removed widgets can shift positional workflow values. The interactive single-image preview is unchanged.

| Widget | Default | Meaning |
|---|---:|---|
| `camera_dolly_m` | 0 | Move the camera closer, 0 to 0.4 m, like the preview scroll wheel. |
| `camera_orbit_yaw_deg` | 0 | Orbit horizontally within a constrained ±10° range; positive looks right. |
| `camera_orbit_pitch_deg` | 0 | Orbit vertically within a constrained ±10° range; positive looks up. |

Orbit uses a fixed pivot on the original viewing axis at the convergence distance for fixed stereo modes with finite convergence, otherwise 1 m. Translation is relative to the resulting camera orientation. Fixed convergence remains measured from the final camera; adjust it if necessary after dollying. Reset camera adjustments restores all three neutral values without resetting stereo settings. Each frame receives the same offsets, with no accumulation or bounding-box-driven orbit pivot. Start with 0.02–0.05 m or a few degrees; the constrained ranges are intended to keep viewpoint changes modest because larger moves can expose missing geometry and increase flicker. Batch zoom and pan were removed to simplify the controls and avoid unhelpful reframing. Test locally before publishing.
