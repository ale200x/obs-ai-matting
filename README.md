# obs-ai-matting — AI background removal for OBS Studio on Linux

![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)
![Platform: Linux](https://img.shields.io/badge/Platform-Linux-informational)
![OBS 32+](https://img.shields.io/badge/OBS-32%2B-302E31)
![GPU: NVIDIA CUDA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900)

**obs-ai-matting** is an open-source **OBS Studio filter for Linux** that removes, blurs or
replaces your webcam background in real time using AI **video matting** on the GPU. Think of
it as a **free, open-source [NVIDIA Broadcast](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/)
alternative for Linux** — a **virtual green screen** that works **without a physical green screen**.

It runs **[Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)** via
**ONNX Runtime + CUDA**, so the cut-out captures fine hair and soft edges and stays stable
frame-to-frame — far cleaner than the usual *segmentation*-based background removers.

> Built and tested on Arch/CachyOS Linux, OBS 32.x, ONNX Runtime 1.24 (CUDA), NVIDIA RTX.

---

## Why it looks better than other OBS background removers

| | Segmentation plugins (MediaPipe / Selfie / SINet) | **obs-ai-matting (RVM)** |
|---|---|---|
| Technique | Per-pixel "person yes/no" mask | **Alpha matting** (continuous 0–1) |
| Edges / hair | Hard, blocky, "cut with scissors" | **Soft, natural, keeps hair** |
| Stability | Flickers, needs heavy smoothing | **Temporally stable** (recurrent model) |
| Quality | Low | **Close to NVIDIA Broadcast** |

The key difference is **matting vs. segmentation**: matting predicts real transparency per
pixel (like a film matte), which is what makes the result look professional.

## Features

- 🟢 **Transparent** mode — outputs the person with alpha; drop any image, **video**, or color
  source *behind* the camera in OBS (native compositing, so any background works).
- 🌫️ **Background blur** mode — built-in, NVIDIA-Broadcast-style blur.
- ⚡ **Threaded GPU inference** — low latency, doesn't stall OBS rendering (~30 fps).
- 🎚️ Per-filter settings: background mode, blur strength, **brightness / gamma** (for low
  light), matte hardness, and quality (384 / 512 / 720).

## Requirements

- OBS Studio 28+ (with `libobs` headers) — tested on 32.x
- ONNX Runtime with the **CUDA** execution provider (e.g. Arch `onnxruntime-opt-cuda`)
- NVIDIA GPU + CUDA + cuDNN (CPU fallback works but is slower)
- CMake and a C++17 compiler

## Build & install

```bash
cmake -B build -S .
cmake --build build
cmake --install build      # -> ~/.config/obs-studio/plugins/obs-ai-matting/
```

## Download the model (required, not bundled)

The RVM model is **not** shipped (it is GPL-3.0 and ~107 MB). Download it once:

```bash
mkdir -p ~/.config/obs-studio/plugins/obs-ai-matting/models
curl -L -o ~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
```

The plugin finds the model via: the **Modelo RVM (.onnx)** field in the filter → the
`$OBS_AI_MATTING_MODEL` env var → `~/.config/obs-studio/plugins/obs-ai-matting/models/` →
`~/ai-camera/models/`.

## Usage

1. Restart OBS.
2. Right-click your camera source → **Filters** → **+** → **AI Background (Matting)**.
3. Choose **Transparente** (then add an image/video/color source *below* the camera for the
   background) or **Desenfoque** (built-in blur).
4. Tune brightness / gamma / hardness / quality.

## FAQ

**Is there a NVIDIA Broadcast for Linux?**
NVIDIA Broadcast itself is Windows-only. obs-ai-matting is an open-source alternative that
gives you AI background removal, blur and a virtual green screen inside OBS Studio on Linux.

**Does it work without a green screen?**
Yes. It's a *virtual* green screen — the AI separates you from any background, no physical
screen or special lighting needed.

**How is it different from the obs-backgroundremoval plugin?**
That plugin mostly relies on lightweight *segmentation* models, which produce hard, blocky
masks. obs-ai-matting uses the **RVM matting** model (continuous alpha + temporal stability),
so edges and hair look much more natural and don't flicker.

**Do I need an NVIDIA GPU?**
It's optimized for NVIDIA + CUDA via ONNX Runtime. It can fall back to CPU, but a GPU is
recommended for real-time use.

**Can I use an image or a video as the background?**
Yes — use **Transparent** mode and place any OBS Image or Media (video) source behind the
camera. OBS composites it for you.

**Is it real-time?**
Yes. Inference runs on a background thread on the GPU at roughly 30 fps at 512px matting.

## How it works

`video_render` captures the source frame (texrender → stage surface → CPU BGRA), applies a
brightness LUT, and hands the frame to a worker thread that runs RVM on CUDA (carrying the
recurrent states for temporal stability). The render thread composites the latest alpha
(≈1 frame latency) — transparent (premultiplied) or blurred — and draws it.

## Contributing

Issues, feature requests and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits & license

- Matting model: **[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)** by
  Peter Lin et al. (GPL-3.0) — downloaded separately.
- Inference: **[ONNX Runtime](https://onnxruntime.ai/)** (MIT).
- This plugin links `libobs`, so it is released under the **GPL-2.0** (see [LICENSE](LICENSE)).

---

<sub>Keywords: OBS Studio background removal Linux, OBS virtual background, OBS background
blur, virtual green screen Linux, NVIDIA Broadcast alternative Linux, AI webcam background,
robust video matting, ONNX Runtime CUDA, real-time portrait matting.</sub>
