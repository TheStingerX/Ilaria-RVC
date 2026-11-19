# Ilaria RVC 2.4.0

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

## Upgrading from 2.3.0

Install the full 2.4.0 ZIP in a new folder. The 2.3.0 updater uses the previous folder layout. Keep your existing installation as a backup while transferring your models and training data.

[Download the Windows ZIP](https://huggingface.co/TheStinger/Ilaria_RVC/resolve/main/releases/2.4.0/ILARIA_RVC-2.4.0.zip?download=true) · [Changelog](docs/CHANGELOG.md) · [License](LICENSE.md) · [Contributing](CONTRIBUTING.md) · [Upstream license](docs/upstream/LICENSE) · [Third-party notices](docs/upstream/THIRD-PARTY-NOTICES.txt)
