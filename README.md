# ♡ Ilaria RVC ♡

Ilaria RVC is a Windows voice conversion and training app based on Mainline RVC.
This repository contains the source for version **1.9.3**.

- Convert one clip, process a batch, or generate speech and convert it with RVC.
- Train voices with live progress, training previews, a dataset checker, and a training queue.
- Import, download, merge, inspect, and export voice models from the **Other** tab.
- Use RMVPE or FCPE for pitch extraction. Version 1.9 also adds streaming local imports and an offline changelog by version.

## Use Ilaria RVC

Get the Windows package from [Ilaria RVC on Hugging Face](https://huggingface.co/TheStinger/Ilaria_RVC).
Extract the Windows package, then open `ILARIA_RVC.bat` for normal use or
`ILARIA_RVC_LOGS.bat` to see the console. The default address is `http://127.0.0.1:6969`.
In an existing installation, open **Other → Update Ilaria RVC** to check for updates.

## Work on the source

For a Windows development setup, start with a clean installation updated to 1.9.3.
Close the app and copy the following items into your source checkout, keeping the same paths:

| Copy from the Windows installation | Purpose |
| --- | --- |
| `runtime/`, `ffmpeg.exe`, `ffprobe.exe` | Python, its dependencies, and audio processing |
| `assets/hubert/hubert_base.pt`, `assets/rmvpe/rmvpe.pt` | ContentVec and RMVPE |
| `assets/pitch/fcpe_runtime/torchfcpe/assets/fcpe_c_v001.pt` | FCPE pitch extraction |
| `assets/pretrained_v2/` | Default RVC v2 training weights, if you need them |

Copy `.env.example` to `.env`, then run `ILARIA_RVC_LOGS.bat`. Models and datasets
belong in `models/` and `datasets/`. The app creates these directories as needed.
An optional `assets/training-completed.wav` sets the completion sound. The source
checkout uses the Windows chime when that file is absent.

The Windows launchers expect the bundled Python layout. The original Mainline
entry points and dependency files are retained for reference in
[the upstream README](README.upstream.md).

## Credits and license

- Based on [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). See the [MIT license](LICENSE) and the original third-party notices.
- Includes adaptations from [Redpanda-vc](https://github.com/redpanda343/Redpanda-vc), with notices in the corresponding source files.
- FCPE uses [TorchFCPE](https://pypi.org/project/torchfcpe/0.0.4/). License notices for vendored Python packages accompany their source.
- The Audio Analyzer extends [Ilaria Audio Analyzer](https://github.com/TheStingerX/Ilaria-Audio-Analyzer).

See [what changed in 1.9.3](CHANGELOG.md).
