# Upstream source inventory

| Component | Upstream | Version / revision | License / attribution |
|---|---|---|---|
| Direct upstream extension | https://github.com/Well-Made/ComfyUI-SHARP-Gaussian-Stereo | fork base; exact revision not recorded here | Original integration and viewer; GPL notices retained |
| SHARP model architecture | https://github.com/PozzettiAndrea/ComfyUI-Sharp | commit `92c0015af61b71e6fd88301cfb336e7034f19796`; Apple-authored vendored model files | Apple Software License; Apple copyright notices retained |
| SHARP model loader and inference integration | https://github.com/PozzettiAndrea/ComfyUI-Sharp | commit `92c0015af61b71e6fd88301cfb336e7034f19796` | GPL-3.0; adapted files identify provenance in module docstrings |
| Original SHARP research code | https://github.com/apple-aiml-research/ml-sharp | upstream repository | Apple Software License; Apple acknowledgements retained |
| Gaussian preview widget and viewer | https://github.com/PozzettiAndrea/comfy-3d-viewers | upstream project version reported as `0.2.58` in prior source inspection | GPL-3.0-or-later; modified viewer files |
| gsplat renderer bundle | https://github.com/dylanebert/gsplat.js | package `gsplat` 1.2.9 | MIT; copyright Dylan Ebert (2023) |

The SHARP checkpoint `sharp_2572gikvuh.pt` is intentionally not included. The loader retrieves it from `https://huggingface.co/apple/Sharp` or reuses a local copy. Its separate Apple Machine Learning Research Model License is included for reference.


## v0.1.2 packaging note
This package was rebuilt from the v0.1.1 release as the base, then updated to include the animated camera controls and explicit COMBO/dropdown UI schema changes.


## Release packaging note
Version 0.1.3 was packaged from the latest validated 0.1.2 Meta Batch / switch-support build and documented as the first releasable consolidated version of this fork.

- v0.2.0: consolidates single-frame inference and browser overlap, optional CuPy covariance and packed transport kernels, WebGL resource updates and bridge wakeups. Two-frame inference experiment removed. Existing upstream licensing unchanged.
