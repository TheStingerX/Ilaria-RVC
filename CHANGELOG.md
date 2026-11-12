# Ilaria RVC Changelog

## 1.1 Beta

- Introduced the purple interface, pink buttons, artwork, favicon, Ko-fi link, and collapsible Training and Inference Settings.

- Numbered the training steps and added progress logs, cumulative steps, estimated finish, and expanded setting ranges.

- Added drag-and-drop inference, the pitch slider, alphabetical voice selection, and automatic index matching.

- Added local and Hugging Face model importing, compatible model merging, and the English Glossary. Fixed Protect values and audio error reporting.

## 1.2.0

- Added the first built-in updater in Other. It checks for releases, verifies downloads, creates backups, restarts the application, and can recover from failed installation.

- Added FP32/FP16 training selection and training performance improvements.

- Added Model information and a hidden artwork tab, cleaned up CMD startup output, and made Single and Batch Inference results easier to read.

## 1.3.0

- Validated automatic updates from 1.2 and showed the installed version in the CMD startup summary.

- Prepared a smaller full Windows release without personal models, datasets, indexes, or UVR files.

## 1.4.0

- Added training previews that convert the same short clip at chosen epoch intervals and keep prior outputs for comparison.

- Added Stop and save, preserving resumable training state and cumulative steps.

- Added Dataset checker for long silences, possible clipping, unreadable audio, and inconsistent levels. Documented it and Training previews in Glossary.

## 1.5.0

- Added drag-and-drop dataset audio and ZIP import directly into datasets. Added Manage datasets and moved Speaker ID below the dataset controls.

- Added the PC training advisor in Other, with hardware-based suggestions. Save frequency defaults to 100 in Training and the advisor.

- Added optional desktop and sound alerts for training completion and errors, a test button, and Ilaria's notification icon.

- Improved contrast, spacing, and borders across Training, Inference, Other, and Glossary.

## 1.5.1

- Process Data now reports a clear error when the dataset path is missing, invalid, empty or contains no nonempty supported audio files directly inside it. No preprocessing job starts in these cases.

- Process Data reports an error if a dataset import is still running; the Glossary explains the dataset checks.

- Renamed Updates to Update Ilaria RVC and added a closed All changes so far section with a concise, deduplicated feature history.

## 1.5.2

- Clearer duplicate-launch message: identifies the installation folder and, when available, the existing launcher PID and Web UI address.

- Explains that closing a browser tab does not stop Ilaria RVC, how to reopen the existing UI, and how to restart without interrupting unsaved training.

- Launcher diagnostics distinguish other lock access errors from an already-running instance; duplicate launches do not stop the original program.

- All four training steps now show a short missing-dataset error before starting; Feature Extraction also checks that Process Data has produced audio.

## 1.6.0

- Training queue: save separate datasets/settings and prepare/train multiple models one after another, with pause, retry and persistent job history.

- Vocoder selection: HiFi-GAN or RefineGAN; this build supports RefineGAN with RVC v2, 32 kHz and Pitch Guidance.

- Embedder Model selection: contentvec or spin-v2, with consistent feature extraction, inference, preview and export metadata.

- Exact cumulative Total Steps targets alongside Epochs, including partial-epoch save/resume and step-based ETA.

- Silent training files slider from 0 to 10, default 2, and selectable Auto/Faiss/KMeans index creation.

- One compact pretrained selector chooses G and D together, with conditional compatible controls and preserved queued pairs.

- Dedicated Hugging Face pretrained downloader plus 35 alphabetical recommended variants with explicit sample rates and exact G/D links.

- Faster pretrained downloads: simultaneous G/D transfers in 16 MiB blocks, independent of UI polling, with speed shown.

- Swift, Beatrice and Parselmouth pitch extraction added to training, single/batch inference and previews; RMVPE remains default.

- Export model ZIP packages only a selected .pth and optional .index; Model information now reads matching FAISS index details.

- Automatic support-folder migration through the existing updater, with verified installation, restart, backup and rollback.

- Clearer spacing, conditional controls and expanded English Glossary; fixed recommended-download percentage formatting and pretrained path handling.

## 1.7.0

Text to speech in Inference

• Open TTS after Single and Batch Inference. Type text or import a UTF-8 .txt file.

• Microsoft Edge’s online engine includes a catalog of 322 voices at launch. Filter by language using names such as Italian and English, or refresh the catalog. Edge sends the text to Microsoft for synthesis.

• Windows voices can generate speech offline using the desktop voices installed on your PC. Both engines offer speaking rate and volume controls. Edge also supports source pitch in Hz.

• RVC conversion is enabled by default and uses the Voice Model selected above the tabs. Its settings include pitch shift, pitch extraction, speaker ID, index, sample rate, volume envelope and consonant protection.

• TTS Result and RVC Result are visible side by side, with TTS+RVC Log below. The converted audio can be downloaded as a WAV directly from its player.

Pink waveform players

• Audio input and output players now show a pink waveform when the browser can decode the file. This includes Single and Batch Inference, TTS and training previews.

• Play and pause from the waveform panel. The playhead moves smoothly and shows elapsed time and total duration. Click or drag to seek, or focus the waveform and use the arrow keys to jump by five seconds.

• Files longer than 30 minutes, larger than 64 MiB or unsupported by the waveform decoder use the browser’s standard player.

Startup and first conversion

• RMVPE loads in the background at startup and is reused during inference. Successful loading stays quiet in the console.

• On supported NVIDIA setups with sufficient free VRAM, the default voice and its embedder prepare in the background. A missing Spin-v2 embedder is downloaded only when requested by normal use.

• Optional libraries load when needed. Gradio startup telemetry is disabled. The Ilaria RVC updater remains available in Other.

• In a local RTX 4070 test with Aerylla, the Web UI became available in about 7 seconds compared with 9.2 seconds previously. After warmup, the first conversion of a one-second clip took about 2 seconds compared with 2.85 seconds. Actual times depend on the PC and the selected model.

## 1.7.1

- Suppressed the specific Windows WinError 10054 traceback during asyncio connection closure. Other connection and conversion errors remain visible.

## 1.7.2

- Fixed conversion when no retrieval index is selected.

- Single and Batch Inference results use the selected model name in the player and the WAV filename, for example Aerylla output.wav.

## 1.8.0

The CMD window is now optional

- ILARIA_RVC.bat opens the Web UI with the console hidden.
- Use ILARIA_RVC_LOGS.bat whenever you want to see the console and logs.
- Other > Reload UI restarts Ilaria RVC and reloads the browser page. It can restart during voice warmup and model loading while protecting active conversions, training and updates.
- Closing the last browser tab shuts down Ilaria RVC and its child processes as soon as the browser reports the closure. This also stops active jobs. Save or finish training first to keep your progress. A browser refresh can send the same signal, so use Reload UI to restart.

Audio Analyzer

- Open Other > Audio Analyzer, directly below Export model ZIP. Compare up to eight local files with two compact tables for file details, levels and silence.
- Inspect the waveform, spectrogram and average spectrum. Estimate pitch is enabled by default.
- Choose a time range, channel view, FFT window, frequency scale and silence threshold. The logarithmic spectrogram now starts at a frequency the selected FFT window can resolve.
- Listen to a preview in the pink player and export the analysis as CSV, JSON or a ZIP with the chart.

Model library

- Inference-ready .pth files and their added indexes share a model folder, such as models/Aerylla/.
- Inference automatically finds the model and its matching index. Training exports and the import, merge and export tools use the new library layout.
- Resumable training checkpoints and trained indexes stay in logs.

## 1.8.1

- Training completion and Test alerts play the Super Mario coin sound. Errors keep the Windows error sound.

## 1.9.0

- Added FCPE to training pitch extraction, training previews, Single and Batch Inference, and TTS. Dataset extraction runs on CPU; inference can use CUDA. RMVPE remains the default.

- Local model imports stream ZIP, PTH and INDEX files in binary, with upload and installation progress, cancellation, and duplicate checks.

- Reorganized Other into Get models, Manage voice models, Audio tools, Training tools, and Application.

- All changes so far now has detailed, expandable notes for each version, available offline.

## 1.9.1

- Fixed a training crash when reusing a Model Name with a different sample rate. Process Data and Train Pth now show the saved and selected rates before making changes. Use a new Model Name to train at a different rate.

## 1.9.2

- Added Google Drive and MEGA to the model downloader in Other. Use a public link to a ZIP, inference .pth or .index file. The downloader handles Google Drive’s large-file confirmation and verifies MEGA downloads before importing them.

## 1.9.3

- Added Split Audio, Autotune, Proposed Pitch, Clean Audio, Formant Shifting and Post-Process effects to Single and Batch Inference. Export in WAV, MP3, FLAC, OGG or M4A.
- Added Embedder Settings below Audio Processing in Single and Batch Inference. Select ContentVec, Spin, Spin-v2, Chinese, Japanese or Korean HuBERT Base, or install a custom HuBERT from a .bin and .json pair.
- Missing Spin and language HuBERT downloads start immediately when the updated program launches, before the interface loads. Each download is verified before installation.
