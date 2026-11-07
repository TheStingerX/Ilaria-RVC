"""Validate selectable training architectures and targets before spawning work."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path
from ilaria_embedders import check_feature_embedder
from ilaria_pretrain_selector import AUTO, validate_pretrain_paths


def validate_options(root, name, rate, pitch, version, vocoder, embedder, silence,
                     target, steps, algorithm, pretrain, generator, discriminator, cache):
    if vocoder not in ('HiFi-GAN', 'RefineGAN'):
        raise ValueError('Choose HiFi-GAN or RefineGAN.')
    if vocoder == 'RefineGAN' and (version != 'v2' or rate != '32k' or not pitch):
        raise ValueError('RefineGAN requires RVC v2, 32 kHz and Pitch Guidance enabled.')
    if vocoder == 'RefineGAN' and pretrain == AUTO:
        raise ValueError('Select a compatible RefineGAN pretrained pair, Custom paths, or None (train from scratch).')
    if embedder not in ('contentvec', 'spin-v2'):
        raise ValueError('Choose contentvec or spin-v2.')
    if isinstance(silence, bool) or int(silence) != silence or not 0 <= int(silence) <= 10:
        raise ValueError('Silent training files must be between 0 and 10.')
    if target not in ('Epochs', 'Steps'):
        raise ValueError('Choose Epochs or Steps.')
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError('Total Steps must be a positive whole number.')
    if target == 'Steps' and cache:
        raise ValueError('Set Cache data to GPU to No when training by Total Steps.')
    if algorithm not in ('Auto', 'Faiss', 'KMeans'):
        raise ValueError('Choose Auto, Faiss or KMeans.')
    if pretrain == AUTO and (not generator or not discriminator):
        raise ValueError('The default pretrained pair is missing. Select an installed pair or explicitly choose None to train from scratch.')
    validate_pretrain_paths(generator, discriminator, root)
    folder = Path(root) / 'logs' / name
    check_feature_embedder(folder, embedder, version)
    if embedder == 'spin-v2' and silence:
        silent = Path(root) / 'logs' / 'mute_spin-v2' / ('3_feature256' if version == 'v1' else '3_feature768') / 'mute.npy'
        if not silent.is_file():
            raise ValueError('Run Feature Extraction with spin-v2 first to prepare its silent features.')
    return {'vocoder': vocoder, 'embedder_model': embedder, 'silent_training_files':int(silence),
            'training_target':target, 'total_steps':int(steps), 'index_algorithm':algorithm}
