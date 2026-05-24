# obs-ai-matting

An OBS Studio filter that removes / blurs / replaces your camera background using
**[Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)** running
on the GPU via ONNX Runtime.

Unlike the usual *segmentation* based background removers (which produce hard, blocky
masks), this uses a **matting** model: it predicts a continuous alpha per pixel, so it
captures hair, soft edges and motion naturally — closer to NVIDIA Broadcast quality. RVM
is recurrent, so the matte is **temporally stable** (no flicker) without aggressive
post-processing.

> Built and tested on Linux (Arch/CachyOS), OBS 32.x, ONNX Runtime 1.24 with the CUDA
> execution provider on an NVIDIA RTX GPU.

## Features

- **Transparent** mode — outputs the person with alpha; put any image/**video**/color
  source *behind* the camera in OBS (native OBS compositing, works with anything).
- **Blur** mode — blurs the background in-place.
- Runs RVM inference on a **worker thread** (low latency, doesn't stall OBS rendering).
- Per-filter settings: background mode, blur strength, **brightness (gain) / gamma**
  (handy for low light), matte hardness, and quality (384 / 512 / 720).

## Requirements

- OBS Studio (with `libobs` dev headers)
- ONNX Runtime with the CUDA execution provider (e.g. Arch `onnxruntime-opt-cuda`)
- CUDA + cuDNN + an NVIDIA GPU (CPU fallback works but is slower)
- CMake, a C++17 compiler

## Build & install

```bash
cmake -B build -S .
cmake --build build
cmake --install build      # -> ~/.config/obs-studio/plugins/obs-ai-matting/
```

## Get the model (required, not bundled)

The RVM model is **not** included (it is GPL-3.0 and ~107 MB). Download `rvm_resnet50_fp32.onnx`
from the RVM releases and place it where the plugin looks for it:

```bash
mkdir -p ~/.config/obs-studio/plugins/obs-ai-matting/models
curl -L -o ~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
```

The plugin resolves the model path in this order:
1. the **Modelo RVM (.onnx)** field in the filter settings,
2. the `$OBS_AI_MATTING_MODEL` environment variable,
3. `~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx`,
4. `~/ai-camera/models/rvm_resnet50.onnx`.

## Usage

1. Restart OBS.
2. Right-click your camera source → **Filters** → **+** → **AI Background (Matting)**.
3. Pick a mode:
   - **Transparente**: add an Image / Media (video) / Color source *below* the camera for the background.
   - **Desenfoque**: built-in background blur.
4. Tune brightness / gamma / hardness / quality to taste.

## How it works

`video_render` captures the source frame (texrender → stage surface → CPU BGRA), applies a
brightness LUT, and hands the frame to a worker thread that runs RVM (CUDA, carrying the
recurrent states for temporal stability). The render thread composites the most recent
alpha (≈1 frame latency) — transparent (premultiplied alpha) or blurred — and draws it.

## Credits & licenses

- Background matting model: **[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)**
  by Peter Lin et al. — GPL-3.0. The model is downloaded separately (see above).
- Inference: [ONNX Runtime](https://onnxruntime.ai/) — MIT.
- This plugin links `libobs` and is therefore released under the **GPL-2.0** (see `LICENSE`).
