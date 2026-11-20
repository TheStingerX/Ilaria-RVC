# Ilaria RVC 2.4.2

The full Windows ZIP is for NVIDIA GPUs and includes Python 3.10, PyTorch 2.10, and CUDA 12.8.

## Getting started

1. Extract the full ZIP to a new folder.
2. Run `RUN_ILARIA_RVC.bat`. To show the console, run `RUN_ILARIA_RVC_LOGS.bat`.
3. Import voice models from the Web UI, or place each model's `.pth` and matching `.index` files in its own subfolder under `models/`.

| Folder | Contents |
| --- | --- |
| `models/` | Voice models and retrieval indexes |
| `app/` | Application, runtime, configuration, and audio assets |
| `app/datasets/` | Audio for training |
| `app/logs/` | Prepared training data, logs, and checkpoints |
| `docs/` | Changelog, feature overview, and upstream documentation |
| `notebooks/` | Kaggle, Google Colab, and Lightning AI Studio notebooks |

## Training precision

Automatic precision uses native BF16 on RTX 30, 40 and 50 series when every selected GPU supports it. Kaggle T4 uses FP16 and other setups default to FP32.

Turn off **Automatic training precision** in Training Settings to choose a supported precision manually. Ilaria RVC remembers this choice. Resuming keeps the saved precision unless you enable **Allow changing precision for saved training**.

BF16 uses mixed precision in the trainer. Feature extraction uses FP32 for BF16 runs, and model exports retain the existing RVC format.

## Updating from 2.4.0 or 2.4.1

Open **Other → Update Ilaria RVC** to install 2.4.2. Kokoro can be installed separately in **Inference → TTS**.

## Upgrading from 2.3.0

Install the full 2.4.0 ZIP in a new folder. The 2.3.0 updater uses the previous folder layout. Keep your existing installation as a backup while transferring your models and training data.

[Download the Windows ZIP](https://huggingface.co/TheStinger/Ilaria_RVC/resolve/main/releases/2.4.0/ILARIA_RVC-2.4.0.zip?download=true) · [Changelog](docs/CHANGELOG.md) · [License](LICENSE.md) · [Contributing](CONTRIBUTING.md) · [Upstream license](docs/upstream/LICENSE) · [Third-party notices](docs/upstream/THIRD-PARTY-NOTICES.txt)
