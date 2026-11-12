"""ContentVec and checksum-verified Spin v2 for the bundled fairseq runtime."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from urllib.request import Request, urlopen

SPIN_URL = 'https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/embedders/fairseq/spin-v2.pt?download=true'
SPIN_SHA256 = '5409a5d5cdfe537cbd2afedcee485d56d402692b5fcf6e39d528fc4b77a693ab'
DOWNLOAD_LOCK = threading.Lock()


def ensure_embedder(root, name):
    root = Path(root)
    if name not in ('contentvec', 'spin-v2'):
        raise ValueError('Choose contentvec or spin-v2.')
    base = root / 'assets' / 'hubert' / 'hubert_base.pt'
    if not base.is_file():
        raise ValueError('Missing assets/hubert/hubert_base.pt. Restore the base embedder first.')
    if name == 'contentvec':
        return
    with DOWNLOAD_LOCK:
        target = root / 'assets' / 'hubert' / 'spin-v2.pt'
        if target.is_file():
            return
        temporary = target.with_suffix('.' + uuid.uuid4().hex + '.partial')
        try:
            yield 'Spin-v2: downloading required embedder (about 361 MiB)...'
            digest, copied = hashlib.sha256(), 0
            with urlopen(Request(SPIN_URL, headers={'User-Agent':'ILARIA-RVC'}), timeout=60) as response, temporary.open('xb') as stream:
                total = int(response.headers.get('Content-Length') or 378877715)
                while True:
                    chunk = response.read(4 * 1024**2)
                    if not chunk: break
                    stream.write(chunk); digest.update(chunk); copied += len(chunk)
                    yield 'Spin-v2 download: %.1f%%' % min(99.9, copied / total * 100)
            if digest.hexdigest() != SPIN_SHA256:
                raise ValueError('Spin-v2 checksum did not match. Retry the download.')
            os.replace(str(temporary), str(target))
            yield 'Spin-v2 embedder ready.'
        finally:
            if temporary.exists(): temporary.unlink()


def load_embedder(root, name='contentvec', device='cpu', half=False):
    from fairseq import checkpoint_utils
    import torch
    root = Path(root)
    for _ in ensure_embedder(root, name): pass
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(root/'assets/hubert/hubert_base.pt')], suffix='')
    model = models[0]
    if name == 'spin-v2':
        checkpoint = torch.load(str(root/'assets/hubert/spin-v2.pt'), map_location='cpu', weights_only=True)
        weights = {key.replace('.parametrizations.weight.original0','.weight_g').replace('.parametrizations.weight.original1','.weight_v'): value
                   for key,value in checkpoint['model'].items()}
        model.load_state_dict(weights, strict=True)
        cfg.task.normalize = bool(checkpoint['cfg']['task'].get('normalize', False))
    model = model.to(device)
    model = model.half() if half else model.float()
    return model.eval(), cfg


def check_feature_embedder(folder, name, version):
    folder = Path(folder)
    manifest = folder / 'ilaria_features.json'
    feature_folder = folder / ('3_feature256' if version == 'v1' else '3_feature768')
    exists = feature_folder.is_dir() and any(feature_folder.glob('*.npy'))
    previous = json.loads(manifest.read_text(encoding='utf-8')).get('embedder_model') if manifest.is_file() else 'contentvec'
    if exists and previous != name:
        raise ValueError('Existing features use %s. Use a new Model Name before switching embedders.' % previous)


def save_feature_embedder(folder, name):
    path = Path(folder) / 'ilaria_features.json'
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps({'embedder_model':name})+'\n',encoding='utf-8')
    os.replace(str(temporary),str(path))


from ilaria_hubert_assets import (
    INFERENCE_EMBEDDERS, custom_embedder_names, install_custom_embedder, inference_embedder_directory,
)


def load_inference_embedder(config, embedder_model="contentvec"):
    root = os.getcwd()
    if embedder_model in ("contentvec", "spin-v2"):
        return load_embedder(root, embedder_model, config.device, config.is_half)[0]
    from ilaria_hf_hubert import load_hf_hubert

    directory = inference_embedder_directory(root, embedder_model)
    return load_hf_hubert(directory, config.device, config.is_half)
