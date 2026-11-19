# Ilaria RVC 2.4.0

Versione Windows per GPU NVIDIA. Nello ZIP completo è incluso il runtime Python 3.10 con PyTorch 2.10 e CUDA 12.8.

## Avvio

1. Estrai lo ZIP in una nuova cartella.
2. Avvia `RUN_ILARIA_RVC.bat` oppure `RUN_ILARIA_RVC_LOGS.bat` per vedere la console.
3. Importa i modelli dalla Web UI, oppure aggiungi i file `.pth` e `.index` alla cartella `models/` organizzandoli per modello.

| Cartella | Contenuto |
| --- | --- |
| `models/` | Libreria di modelli vocali e index |
| `app/` | Programma, configurazioni, dipendenze e risorse audio |
| `app/datasets/` | Audio per il training |
| `app/logs/` | Dati di training, log e checkpoint |
| `docs/` | Changelog, funzionalità e documentazione upstream |
| `notebooks/` | Kaggle, Google Colab e Lightning AI Studio |

Per passare dalla 2.3.0 a questa struttura usa lo ZIP completo in una nuova cartella. I vecchi aggiornamenti automatici usano il layout precedente. Conserva l'installazione precedente mentre trasferisci i dati.

[Changelog](docs/CHANGELOG.md) · [Licenza](LICENSE.md) · [Come contribuire](CONTRIBUTING.md) · [Licenza upstream](docs/upstream/LICENSE) · [Licenze di terze parti](docs/upstream/THIRD-PARTY-NOTICES.txt)
