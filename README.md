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

**New in 0.2**: 💡 **auto light match** — the subject's exposure and white balance gently
follow the background you put behind it in OBS, so you actually look like you're *in* the
scene instead of pasted on top of it.

<p align="center">
  <img src="assets/lightmatch-grid.jpg" alt="Auto light match demo — same camera and pose across three backgrounds: the subject cools down in a blue sci-fi room, warms up in a red neon tunnel, and brightens in a white office" width="100%">
  <br>
  <sub><b>Auto light match in action</b> — same camera, same pose, real time: skin tone cools
  down in the blue sci-fi room, picks up the warm cast in the neon tunnel, and brightens in
  the white office. (<a href="assets/lightmatch-scifi-blue.jpg">full-size 1</a> ·
  <a href="assets/lightmatch-neon-warm.jpg">2</a> ·
  <a href="assets/lightmatch-office-bright.jpg">3</a>)</sub>
</p>

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
- 💡 **Auto light match** — samples the background source you picked in OBS and gently
  adjusts the subject's exposure and white balance so the cut-out **looks lit by the scene**
  (subtle by design; strength slider; temporally smoothed, no flicker).
- ⚡ **Threaded GPU inference** — low latency, doesn't stall OBS rendering (~30 fps).
- 🎚️ Per-filter settings: background mode, blur strength, **brightness / gamma** (for low
  light), matte hardness, and quality (384 / 512 / 720).
- 🌐 **Localized UI** — English and Spanish, following your OBS language automatically.

## Requirements

- OBS Studio 28+ (with `libobs` headers) — tested on 32.x
- ONNX Runtime with the **CUDA** execution provider (e.g. Arch `onnxruntime-opt-cuda`)
- NVIDIA GPU + CUDA + cuDNN (if CUDA is unavailable or fails to initialize, the plugin
  falls back to CPU instead of disabling the matte)
- CMake and a C++17 compiler

## Install on Arch Linux (recommended)

```bash
git clone https://github.com/ale200x/obs-ai-matting.git
cd obs-ai-matting/packaging/arch
makepkg -si
```

The package pins the **ONNX Runtime series it was built against** on purpose — see
[the filter disappeared after a system update](#the-filter-disappeared-after-a-system-update)
for why that matters. Upgrading ONNX Runtime will then ask you to rebuild the plugin
instead of silently breaking it.

## Build & install manually

```bash
cmake -B build -S .
cmake --build build
cmake --install build      # -> ~/.config/obs-studio/plugins/obs-ai-matting/
```

⚠️ A manual install is **not tracked by your package manager**: when ONNX Runtime is
upgraded you must re-run those three commands yourself, or the filter stops loading
(again, see [below](#the-filter-disappeared-after-a-system-update)).

## Download the model (required, not bundled)

The RVM model is **not** shipped (it is GPL-3.0 and ~107 MB). Download it once:

```bash
mkdir -p ~/.config/obs-studio/plugins/obs-ai-matting/models
curl -L -o ~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
```

If you installed the **package**, put it where the plugin looks for it outside the
home-layout instead:

```bash
mkdir -p ~/.local/share/obs-ai-matting/models
curl -L -o ~/.local/share/obs-ai-matting/models/rvm_resnet50.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
```

The plugin finds the model via: the **Modelo RVM (.onnx)** field in the filter → the
`$OBS_AI_MATTING_MODEL` env var → `~/.config/obs-studio/plugins/obs-ai-matting/models/` →
`$XDG_DATA_HOME/obs-ai-matting/models/` → `/usr/share/obs-ai-matting/models/` →
`~/ai-camera/models/`.

## Usage

1. Restart OBS.
2. Right-click your camera source → **Filters** → **+** → **AI Background (Matting)**.
3. Choose **Transparent** (then add an image/video/color source *below* the camera for the
   background) or **Blur** (built-in blur).
4. Tune brightness / gamma / hardness / quality.
5. *(Optional)* Enable **Match lighting to background (auto)** and pick your background
   source (or the whole scene) in **Background source** — the subject's light will subtly
   follow the background. **Match strength** controls how strong the match is.

<p align="center">
  <img src="assets/filter-settings.png" alt="Filter settings panel in OBS with auto light match enabled: background mode, blur strength, match lighting checkbox, background source, match strength, brightness, gamma and matte hardness" width="70%">
  <br>
  <sub>The filter's settings with <b>auto light match</b> enabled — while running at
  <b>60 fps with ~13% CPU</b> (laptop RTX 4050, 512 px matting).</sub>
</p>

## Troubleshooting

### The filter disappeared after a system update

**Symptom:** *"AI Background (Matting)"* is gone from the filter list — and OBS also
**removed the filter from your scene**, along with all of its settings.

**Check the OBS log** (`Help → Log Files → Show Log Files`, or
`~/.config/obs-studio/logs/`) for this:

```
os_dlopen(...obs-ai-matting.so): /usr/lib/libonnxruntime.so.1:
  version `VERS_1.28.0' not found (required by ...obs-ai-matting.so)
Module '...obs-ai-matting.so' not loaded
Source ID 'obs_ai_matting' not found
Failed to create source 'AI Background (Matting)'!
```

**Cause:** ONNX Runtime exports **versioned symbols** (`VERS_1.28.0`, `VERS_1.29.0`, …)
and bumps them on every minor release **without changing the soname** — it stays
`libonnxruntime.so.1`. So nothing looks broken from the outside: the library is there,
the soname matches, but the plugin was linked against symbols the new build no longer
exports. It stops loading. And because OBS can't resolve the source ID, it drops the
filter from the scene the next time it saves — that's why your settings vanish too.

The same thing happens if `obs-studio` bumps the `libobs` soname.

**Fix** — rebuild it against the current libraries:

```bash
cmake --build build && cmake --install build   # manual install
# or, if you installed the package:
cd packaging/arch && makepkg -si
```

Then **restart OBS** and add the filter to your camera source again.

**Avoid it:** install the **[Arch package](#install-on-arch-linux-recommended)**. It pins
the ONNX Runtime series it was built against, so the upgrade asks you to rebuild the
plugin instead of leaving you with a module that no longer loads. If you build manually,
rebuild the plugin **before** opening OBS after an ONNX Runtime upgrade — once OBS opens
with a broken module, the filter (and its settings) are already gone from the scene.

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

**Can the subject's lighting match the background?**
Yes — enable **auto light match** and select the background source (a scene works too: the
plugin measures "the scene without you"). It nudges exposure and white balance toward the
background's average light — a warm background warms you up slightly, a dark one dims you a
bit — with tight clamps so you always stay readable and never get tinted.

**Is it real-time?**
Yes. Inference runs on a background thread on the GPU at roughly 30 fps at 512px matting.
If the CUDA provider cannot start, the plugin automatically falls back to CPU and writes a
warning to the OBS log. CPU performance depends on the processor; use 384 or 512 quality
instead of 720 for a more responsive preview.

## How it works

`video_render` captures the source frame (texrender → stage surface → CPU BGRA), applies a
brightness LUT, and hands the frame to a worker thread that runs RVM on CUDA (carrying the
recurrent states for temporal stability). The render thread composites the latest alpha
(≈1 frame latency) — transparent (premultiplied) or blurred — and draws it.

**Auto light match**: every 15 frames the filter renders the selected background source at
64×36 (GPU downscale) and takes its alpha-weighted mean color; the worker computes the
subject's mean color (alpha-weighted, on the small inference buffers). From both means it
derives partial-exposure (`(Yb/Yf)^0.55`, clamped) and white-balance per-channel gains,
smoothed with an EMA and baked into per-channel LUTs applied at composition time. The
subject stats are taken *before* the auto adjustment, so there is no feedback loop. If the
background is a scene containing the camera itself, a re-entrancy guard makes the camera
contribute nothing to the sample — the measurement is exactly "the scene without you".

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
robust video matting, ONNX Runtime CUDA, real-time portrait matting, auto light match,
match webcam lighting to background, relight webcam OBS.</sub>
