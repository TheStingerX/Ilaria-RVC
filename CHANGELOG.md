# ILARIA RVC 1.9.2

- Added Google Drive and MEGA to the model downloader in Other. Use a public link to a ZIP, inference .pth or .index file. The downloader handles Google Drive’s large-file confirmation and verifies MEGA downloads before importing them.

# ILARIA RVC 1.9.1

- Fixed a training crash when reusing a Model Name with a different sample rate. Process Data and Train Pth now show the saved and selected rates before making changes. Use a new Model Name to train at a different rate.

# ILARIA RVC 1.9.0

- Added FCPE to training pitch extraction, training previews, Single and Batch Inference, and TTS. Dataset extraction runs on CPU; inference can use CUDA. RMVPE remains the default.

- Local model imports stream ZIP, PTH and INDEX files in binary, with upload and installation progress, cancellation, and duplicate checks.

- Reorganized Other into Get models, Manage voice models, Audio tools, Training tools, and Application.

- All changes so far now has detailed, expandable notes for each version, available offline.
