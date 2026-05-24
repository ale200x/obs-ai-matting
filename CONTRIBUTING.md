# Contributing

Thanks for your interest in improving **obs-ai-matting**! Contributions of all kinds are
welcome — bug reports, ideas, docs and code.

## Reporting bugs / requesting features

Open an [issue](https://github.com/ale200x/obs-ai-matting/issues) and include:

- Your distro, OBS version, GPU + driver, ONNX Runtime version.
- What you expected vs. what happened (the OBS log helps: `Help → Log Files`).
- Steps to reproduce.

For questions or ideas, use [Discussions](https://github.com/ale200x/obs-ai-matting/discussions).

## Submitting changes (pull requests)

1. **Fork** the repo and create a branch (`git checkout -b my-fix`).
2. Build and test locally (`cmake -B build -S . && cmake --build build`), confirm the filter
   loads and works in OBS.
3. Keep the style consistent with the existing code (tabs, C++17).
4. Open a **Pull Request** describing the change and why.

## Ideas / roadmap

- GPU compositing (shader) instead of CPU.
- Bundle/AUR packaging and CI build.
- More background modes (color key, image fit options).
- Optional models (e.g. lighter RVM mobilenetv3 for low-end GPUs).

## License

By contributing you agree your changes are licensed under the project's **GPL-2.0**.
