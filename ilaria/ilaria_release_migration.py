"""Transactional support-folder migration for legacy-compatible update ZIPs."""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

STATE = '.ilaria-updates/layout-migration.json'
LEGACY_PAYLOAD = 'ilaria_release_payload.py'
NESTED_PAYLOAD = 'ilaria/ilaria_release_payload.py'
EXTRA_FILES = frozenset(('infer/lib/audio.py', 'infer/modules/vc/pipeline.py', 'infer/modules/vc/utils.py', 'ilaria/tts_voices.json', 'ilaria/tts_vendor/certifi/__init__.py', 'ilaria/tts_vendor/certifi/__main__.py', 'ilaria/tts_vendor/certifi/cacert.pem', 'ilaria/tts_vendor/certifi/core.py', 'ilaria/tts_vendor/certifi/py.typed', 'ilaria/tts_vendor/certifi/tests/__init__.py', 'ilaria/tts_vendor/certifi/tests/test_certify.py', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/INSTALLER', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/licenses/LICENSE', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/METADATA', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/RECORD', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/REQUESTED', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/top_level.txt', 'ilaria/tts_vendor/certifi-2026.10.23.dist-info/WHEEL', 'ilaria/tts_vendor/edge_tts/__init__.py', 'ilaria/tts_vendor/edge_tts/__main__.py', 'ilaria/tts_vendor/edge_tts/communicate.py', 'ilaria/tts_vendor/edge_tts/constants.py', 'ilaria/tts_vendor/edge_tts/data_classes.py', 'ilaria/tts_vendor/edge_tts/drm.py', 'ilaria/tts_vendor/edge_tts/exceptions.py', 'ilaria/tts_vendor/edge_tts/progress.py', 'ilaria/tts_vendor/edge_tts/py.typed', 'ilaria/tts_vendor/edge_tts/srt_composer.py', 'ilaria/tts_vendor/edge_tts/submaker.py', 'ilaria/tts_vendor/edge_tts/typing.py', 'ilaria/tts_vendor/edge_tts/util.py', 'ilaria/tts_vendor/edge_tts/version.py', 'ilaria/tts_vendor/edge_tts/voices.py', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/entry_points.txt', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/INSTALLER', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/licenses/LICENSE', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/METADATA', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/RECORD', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/REQUESTED', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/top_level.txt', 'ilaria/tts_vendor/edge_tts-7.3.1.dist-info/WHEEL', 'ILARIA_RVC_LOGS.bat', 'infer/lib/train/process_ckpt.py', 'assets/training-completed.wav', 'assets/pitch/fcpe_runtime/einops/array_api.py', 'assets/pitch/fcpe_runtime/einops/einops.py', 'assets/pitch/fcpe_runtime/einops/packing.py', 'assets/pitch/fcpe_runtime/einops/parsing.py', 'assets/pitch/fcpe_runtime/einops/py.typed', 'assets/pitch/fcpe_runtime/einops/_backends.py', 'assets/pitch/fcpe_runtime/einops/_torch_specific.py', 'assets/pitch/fcpe_runtime/einops/__init__.py', 'assets/pitch/fcpe_runtime/einops/experimental/indexing.py', 'assets/pitch/fcpe_runtime/einops/experimental/__init__.py', 'assets/pitch/fcpe_runtime/einops/layers/flax.py', 'assets/pitch/fcpe_runtime/einops/layers/keras.py', 'assets/pitch/fcpe_runtime/einops/layers/oneflow.py', 'assets/pitch/fcpe_runtime/einops/layers/paddle.py', 'assets/pitch/fcpe_runtime/einops/layers/tensorflow.py', 'assets/pitch/fcpe_runtime/einops/layers/torch.py', 'assets/pitch/fcpe_runtime/einops/layers/_einmix.py', 'assets/pitch/fcpe_runtime/einops/layers/__init__.py', 'assets/pitch/fcpe_runtime/einops/tests/run_tests.py', 'assets/pitch/fcpe_runtime/einops/tests/test_einsum.py', 'assets/pitch/fcpe_runtime/einops/tests/test_examples.py', 'assets/pitch/fcpe_runtime/einops/tests/test_layers.py', 'assets/pitch/fcpe_runtime/einops/tests/test_ops.py', 'assets/pitch/fcpe_runtime/einops/tests/test_other.py', 'assets/pitch/fcpe_runtime/einops/tests/test_packing.py', 'assets/pitch/fcpe_runtime/einops/tests/test_parsing.py', 'assets/pitch/fcpe_runtime/einops/tests/__init__.py', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/INSTALLER', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/METADATA', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/RECORD', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/REQUESTED', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/WHEEL', 'assets/pitch/fcpe_runtime/einops-0.8.2.dist-info/licenses/LICENSE', 'assets/pitch/fcpe_runtime/local_attention/local_attention.py', 'assets/pitch/fcpe_runtime/local_attention/rotary.py', 'assets/pitch/fcpe_runtime/local_attention/transformer.py', 'assets/pitch/fcpe_runtime/local_attention/__init__.py', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/INSTALLER', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/LICENSE', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/METADATA', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/RECORD', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/REQUESTED', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/top_level.txt', 'assets/pitch/fcpe_runtime/local_attention-1.9.15.dist-info/WHEEL', 'assets/pitch/fcpe_runtime/torchfcpe/mel_extractor.py', 'assets/pitch/fcpe_runtime/torchfcpe/mel_fn_librosa.py', 'assets/pitch/fcpe_runtime/torchfcpe/models.py', 'assets/pitch/fcpe_runtime/torchfcpe/models_infer.py', 'assets/pitch/fcpe_runtime/torchfcpe/model_conformer_naive.py', 'assets/pitch/fcpe_runtime/torchfcpe/model_convnext.py', 'assets/pitch/fcpe_runtime/torchfcpe/tools.py', 'assets/pitch/fcpe_runtime/torchfcpe/torch_interp.py', 'assets/pitch/fcpe_runtime/torchfcpe/__init__.py', 'assets/pitch/fcpe_runtime/torchfcpe/assets/fcpe_c_v001.pt', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/INSTALLER', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/LICENSE', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/METADATA', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/RECORD', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/REQUESTED', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/top_level.txt', 'assets/pitch/fcpe_runtime/torchfcpe-0.0.4.dist-info/WHEEL'))

def _support_module(name):
    return bool(re.fullmatch(r'ilaria/ilaria_[a-z0-9_]+\.(?:py|json)', name))

def _payload_file(name):
    return name in EXTRA_FILES or _support_module(name)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, name):
    candidate = root / name
    if candidate.resolve() != candidate or root not in candidate.parents:
        raise ValueError('The migration folder must not contain links or junctions.')
    return candidate


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists(): temporary.unlink()


def _save(root, state):
    _write(_path(root, STATE), (json.dumps(state, indent=2)+'\n').encode('utf-8'))


def _state(root):
    path = _path(root, STATE)
    if not path.is_file(): return None
    state = json.loads(path.read_text(encoding='utf-8'))
    if not re.fullmatch('[a-f0-9]{32}', state.get('id','')):
        raise ValueError('Invalid layout backup identifier.')
    for name in state['installed']:
        if not _payload_file(name):
            raise ValueError('Invalid support module in layout backup.')
    return state


def install_layout(root, files, version, managed_root):
    root = Path(root).resolve()
    installed = {name:_hash(data) for name,data in files.items()}
    previous = _state(root)
    if previous and previous['version']==version and previous['installed']==installed and previous['phase'] in ('pending','ready'):
        if all(_path(root,name).is_file() and _hash(_path(root,name).read_bytes())==digest for name,digest in installed.items()):
            return
        if previous['phase']=='pending':
            rollback_layout(root)
            previous = _state(root)
        else:
            raise ValueError('Installed migration files changed. Restore their update backup before retrying.')

    if previous and previous['phase'] == 'pending':
        raise ValueError('Another layout update is still pending.')
    identity = uuid.uuid4().hex
    state = {'id':identity,'version':version,'phase':'pending','installed':installed,
             'original':{},'root_files':{},'root_archive':{},'managed_root':list(managed_root)}
    backup = '.ilaria-updates/layout-backups/'+identity+'/'
    for name,data in files.items():
        if not _payload_file(name):
            raise ValueError('Unsupported migration payload path.')
        if name.endswith('.py'): compile(data,name,'exec')
        target = _path(root,name)
        original = target.read_bytes() if target.is_file() else None
        state['original'][name] = _hash(original) if original is not None else None
        if original is not None: _write(_path(root,backup+'before/'+name),original)
    for name in managed_root:
        path = _path(root,name)
        if path.is_file(): state['root_files'][name] = _hash(path.read_bytes())
    # Preserve old nonmanifest helpers as well; old UI versions can need them on rollback.
    for name in files:
        if not _support_module(name): continue
        filename = Path(name).name
        path = _path(root,filename)
        if path.is_file():
            data=path.read_bytes()
            _write(_path(root,backup+'root/'+filename),data)
            state['root_archive'][filename]=_hash(data)
    state['previous'] = previous
    _save(root,state)
    try:
        for name in sorted(files,key=lambda value:value.endswith('ilaria_version.json')):
            _write(_path(root,name),files[name])
            if _hash(_path(root,name).read_bytes())!=installed[name]:
                raise ValueError('Support module verification failed: '+name)
    except Exception:
        rollback_layout(root)
        raise


def installed_file_matches(root, name, expected):
    root=Path(root).resolve();state=_state(root)
    if not state or state['phase']!='ready' or state['root_files'].get(name)!=expected:
        return False
    if name in (LEGACY_PAYLOAD, NESTED_PAYLOAD):
        return all(_path(root,key).is_file() and _hash(_path(root,key).read_bytes())==digest for key,digest in state['installed'].items())
    nested='ilaria/'+name
    return nested in state['installed'] and _path(root,nested).is_file() and _hash(_path(root,nested).read_bytes())==state['installed'][nested]


def rollback_layout(root, expected_payload=None):
    root=Path(root).resolve();state=_state(root)
    if not state or state['phase'] not in ('pending','ready'): return
    if expected_payload is not None and expected_payload not in (
        state['root_files'].get(LEGACY_PAYLOAD), state['root_files'].get(NESTED_PAYLOAD)
    ): return
    backup='.ilaria-updates/layout-backups/'+state['id']+'/'
    # Verify backups and current files before changing anything.
    for name,original in state['original'].items():
        target=_path(root,name)
        if target.is_file() and _hash(target.read_bytes()) not in (original,state['installed'][name]):
            raise ValueError('A support file changed after the update; automatic restore kept it: '+name)
        if original is not None:
            path=_path(root,backup+'before/'+name)
            if not path.is_file() or _hash(path.read_bytes())!=original:raise ValueError('Missing layout backup: '+name)
    for name,original in state['original'].items():
        target=_path(root,name)
        if original is None:
            if target.is_file():target.unlink()
        else:_write(target,_path(root,backup+'before/'+name).read_bytes())
    for name,digest in state['root_archive'].items():
        if name in state['managed_root']:continue
        data=_path(root,backup+'root/'+name).read_bytes()
        if _hash(data)!=digest:raise ValueError('A legacy helper backup is damaged.')
        _write(_path(root,name),data)
    if state.get('previous') is not None:
        _save(root,state['previous'])
    else:
        state['phase']='rolled_back';_save(root,state)


def cleanup(root, keep_updater=False):
    root=Path(root).resolve();state=_state(root)
    if not state or state['phase']!='ready':return
    names=dict(state['root_archive']);names.update(state['root_files'])
    for name,digest in names.items():
        if name in ('ILARIA_RVC.py','ILARIA_RVC.bat') or ('/' in name and name != NESTED_PAYLOAD):continue
        if keep_updater and name=='ilaria_updater.py':continue
        if name not in (LEGACY_PAYLOAD, NESTED_PAYLOAD) and not name.startswith('ilaria_'):continue
        path=_path(root,name)
        if path.is_file() and _hash(path.read_bytes())==digest:path.unlink()


def mark_ready(root, keep_updater=False):
    root=Path(root).resolve();state=_state(root)
    if not state:return
    if state['phase']=='pending':
        if not all(_path(root,key).is_file() and _hash(_path(root,key).read_bytes())==digest for key,digest in state['installed'].items()):
            raise ValueError('Support-folder migration is incomplete.')
        state['phase']='ready';_save(root,state)
    # Cosmetic cleanup must not turn a usable, verified app into a failed start.
    try:cleanup(root,keep_updater)
    except OSError:pass
