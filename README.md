# ♡ Ilaria RVC ♡

Ilaria RVC is a Windows voice conversion and training app based on Mainline RVC.
This repository contains the source for version **2.0.0**.

## Features added from Mainline RVC

### Inference and text to speech

- Drag and drop audio in **Single** and **Batch Inference**. Models are listed alphabetically, and the app looks up the matching index automatically. The pitch slider ranges from −24 to +24 semitones. Index Ratio starts at 0.
- Export in **WAV, MP3, FLAC, OGG, or M4A**. Outputs use the selected model name, and conversion logs summarize the result.
- **Audio Processing** includes Split Audio, Autotune, Proposed Pitch, Clean Audio, Formant Shifting, and Post-Process effects.
- **Embedder Settings** supports ContentVec, Spin, Spin-v2, Chinese, Japanese and Korean HuBERT Base, and custom HuBERT embedders supplied as a .bin and .json pair. Download optional embedders individually from **Other**.
- Pitch extraction includes **FCPE, Swift, Beatrice, and Parselmouth**, with RMVPE as the default. Available choices depend on the operation.
- **TTS** accepts typed text or UTF-8 text files. Use Microsoft Edge online voices or installed Windows voices offline. Filter voices by language and adjust rate, volume, and source pitch where supported. Listen to the source speech and RVC conversion side by side.
- Pink waveform players support playback, seeking, and keyboard controls for supported audio across inference, TTS, and training previews.

### Training and datasets

- Numbered training workflow with collapsible settings, compact progress logs, cumulative steps, estimated finish time, and completion messages.
- Train to an **epoch or exact total-step target**, with partial-epoch resume for step targets. **Stop and save** writes a resumable checkpoint. The persistent **Training Queue** can prepare and train multiple models with pause and retry controls.
- Choose **FP32 or FP16**, ContentVec or Spin-v2, HiFi-GAN or RefineGAN, Auto/Faiss/KMeans index creation, and the number of silent training files. RefineGAN training supports RVC v2 at 32 kHz with pitch guidance.
- Select generator and discriminator pretrained weights together. The pretrained downloader includes a catalog of compatible variants with sample rates and downloads both weights in parallel.
- Import dataset audio and ZIPs by drag and drop and manage imported datasets in the UI. All four training steps check for missing datasets. Training also checks sample-rate compatibility before reusing a model folder.
- **Dataset Checker** flags long silences, possible clipping, unreadable files, and inconsistent levels. Listen to a 12-second preview at a chosen timestamp or near the longest silence.
- **Training previews** convert the same clip at selected epoch intervals and retain earlier outputs for comparison. Optional desktop and sound alerts report training completion or errors.
- **Training Dashboard** displays epoch duration, cumulative steps, steps per second, and loss charts. Refresh runs in the background while you switch tabs. **TensorBoard** opens inside the dashboard.
- **PC Training Advisor** suggests settings based on your hardware. **Diagnostic Report** provides a copyable summary of Windows version, GPU, CUDA, free disk space, and recent training errors.

### Models and audio tools

- **Other** downloads public voice model ZIP, PTH, and INDEX files from Hugging Face, Google Drive, and MEGA.
- Local ZIP, PTH, and INDEX imports stream directly with progress, cancellation, and duplicate checks.
- Inference-ready models and their indexes share a folder under **models/**. Training exports and model management tools use the same library.
- Merge compatible models with adjustable proportions. **Model information** shows model and matching FAISS index details. **Export model ZIP** packages the selected model and optional index for sharing.
- **Audio Analyzer** compares up to eight files, with waveform, spectrogram, average spectrum, pitch estimation, and level and silence measurements. Preview a range and export results as CSV, JSON, or a ZIP with charts.

### Interface, startup, and updates

- Purple interface with pink controls, custom artwork and favicon, a Ko-fi link, and an English **Glossary** covering settings and workflows.
- Launch with the console hidden or use the separate logs launcher. The console displays version and startup details. Duplicate-launch diagnostics identify an instance already running from the same folder.
- **Restart Ilaria RVC** restarts the app from Other. Closing the last browser tab shuts down the app and its child processes. Save or finish training before closing it.
- RMVPE and, on supported NVIDIA setups, the default voice prepare in the background. Optional libraries load when needed.
- **Update Ilaria RVC** checks for new releases, verifies update downloads, and supports restart, backup, and recovery. Expand **All changes so far** for offline notes by version.

## Use Ilaria RVC

Get the Windows package from [Ilaria RVC on Hugging Face](https://huggingface.co/TheStinger/Ilaria_RVC).
Extract the Windows package, then open `ILARIA_RVC.bat` for normal use or
`ILARIA_RVC_LOGS.bat` to see the console. The default address is `http://127.0.0.1:6969`.
In an existing installation, open **Other → Update Ilaria RVC** to check for updates.

## Work on the source

For a Windows development setup, start with a clean installation updated to 2.0.0.
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
- Includes adaptations from [Applio](https://github.com/IAHispano/Applio) and [Redpanda-vc](https://github.com/redpanda343/Redpanda-vc), with notices in the corresponding source files.
- FCPE uses [TorchFCPE](https://pypi.org/project/torchfcpe/0.0.4/). License notices for vendored Python packages accompany their source.
- The Audio Analyzer extends [Ilaria Audio Analyzer](https://github.com/TheStingerX/Ilaria-Audio-Analyzer).

See [what changed in 2.0.0](CHANGELOG.md).
