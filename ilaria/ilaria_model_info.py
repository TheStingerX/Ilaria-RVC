"""Read-only, CPU-only information about installed RVC inference exports."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re

CONFIG_FIELDS = (
    'Spectrogram channels', 'Export segment frames', 'Inter channels',
    'Hidden channels', 'Filter channels', 'Attention heads', 'Attention layers',
    'Kernel size', 'Dropout', 'ResBlock type', 'ResBlock kernel sizes',
    'ResBlock dilation sizes', 'Upsample rates', 'Initial upsample channels',
    'Upsample kernel sizes', 'Configured speaker slots', 'Speaker embedding channels',
    'Sample rate (Hz)',
)
UNKNOWN = 'Not stored in this file'


from ilaria_model_paths import voice_files, voice_path


def list_model_choices(root):
    return sorted(voice_files(Path(root) / "models"), key=str.casefold)


def _path(name, root):
    return voice_path(Path(root) / "models", name)


def _safe(value, depth=0):
    """Bound exported metadata without evaluating objects or retaining tensors."""
    if depth > 6:
        return '[Nested value omitted]'
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:8000] + (' [truncated]' if len(value) > 8000 else '')
    if isinstance(value, (list, tuple)):
        result = [_safe(item, depth + 1) for item in value[:100]]
        if len(value) > 100:
            result.append('[Further items omitted]')
        return result
    if isinstance(value, dict):
        result = {}
        for i, (key, item) in enumerate(value.items()):
            if i >= 100:
                result['[Further metadata omitted]'] = True
                break
            if isinstance(key, (str, int, float, bool)):
                result[str(key)[:200]] = _safe(item, depth + 1)
        return result
    return '[Non-primitive value omitted]'


def _md(value):
    value = _safe(value)
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(_safe(value), ensure_ascii=True)
    text = html.escape(str(value), quote=True)
    return (text.replace('\\', '\\\\').replace('|', '\\|')
            .replace('`', '\\`').replace('*', '\\*').replace('_', '\\_')
            .replace('[', '\\[').replace(']', '\\]').replace('\n', '<br>'))


def _rate(value):
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in ('32k', '40k', '48k'):
            return int(normalized[:-1]) * 1000
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value == int(value) and value > 0:
            return int(value)
    return None


def _first(model, names):
    for name in names:
        if name in model:
            return _safe(model[name]), name
    return UNKNOWN, None


def _indexes(path, root):
    models = (Path(root) / 'models').resolve()
    if not models.is_dir():
        return []
    matches = []
    stem = path.stem.casefold()
    group = path.parent.name.casefold()
    suffix = re.compile(r'(?:^|_)' + re.escape(stem) + r'(?:_v[12])?$', re.I)
    for candidate in models.rglob('*.index'):
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(models)
        except ValueError:
            continue
        if not resolved.is_file() or candidate.name.casefold().startswith('trained_'):
            continue
        folder_match = bool(relative.parts and len(relative.parts) > 1 and relative.parts[0].casefold() == group)
        if folder_match or suffix.search(candidate.stem):
            matches.append({'path': 'models/' + relative.as_posix(),
                            'size_bytes': resolved.stat().st_size,
                            'match': 'Matching model folder' if folder_match else 'Matching filename'})
        if len(matches) == 100:
            break
    return sorted(matches, key=lambda item: item['path'].casefold())


def _index_details(item, root, expected_dimension):
    """Read FAISS metadata without reconstructing vectors or running a search."""
    details = dict(item)
    try:
        path = (Path(root) / item['path']).resolve()
        before = path.stat()
        details['last_modified_pc_local'] = datetime.fromtimestamp(before.st_mtime).strftime('%d-%m-%Y %H:%M:%S')
        details['size_mib'] = round(before.st_size / 1024**2, 3)
        if before.st_size == 0:
            raise ValueError('The index file is empty.')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
                digest.update(chunk)
        details['sha256'] = digest.hexdigest()
        import faiss
        # IVF lists can be enormous; only their metadata is needed here.
        flags = getattr(faiss, 'IO_FLAG_SKIP_IVF_DATA', 0) | getattr(faiss, 'IO_FLAG_READ_ONLY', 0)
        try:
            index = faiss.read_index(str(path), flags)
        except RuntimeError as error:
            # The bundled Windows FAISS 1.7.2 lacks the IVF I/O hooks.
            if 'InvertedListsIOHook' not in str(error):
                raise
            index = faiss.read_index(str(path))
        dimension, vectors = int(index.d), int(index.ntotal)
        details.update({'faiss_type': type(index).__name__, 'feature_dimension': dimension,
                        'indexed_vectors': vectors, 'is_trained': bool(index.is_trained),
                        'rvc_version_suggested_by_dimension': {256: 'v1', 768: 'v2'}.get(dimension, 'Unknown'),
                        'expected_model_dimension': expected_dimension,
                        'dimension_matches_model': dimension == expected_dimension if expected_dimension else None})
        metric = int(index.metric_type)
        details['distance_metric'] = {getattr(faiss, 'METRIC_L2', 1): 'L2 (squared Euclidean distance)',
                                      getattr(faiss, 'METRIC_INNER_PRODUCT', 0): 'Inner product'}.get(metric, 'FAISS metric %s' % metric)
        try:
            ivf = faiss.extract_index_ivf(index)
            details['cluster_count_nlist'] = int(ivf.nlist)
            details['saved_search_clusters_nprobe'] = int(ivf.nprobe)
            details['code_bytes_per_vector'] = int(ivf.code_size)
        except RuntimeError:
            pass  # Flat indexes do not use IVF clusters.
        notes = []
        if not index.is_trained:
            notes.append('This index has not been trained and is not ready for retrieval.')
        if vectors == 0:
            notes.append('No feature vectors are stored; this index cannot provide retrieval results.')
        if details['dimension_matches_model'] is False:
            notes.append('Feature dimension does not match this voice model. Do not use this index with it.')
        details['checks'] = notes or ['Header checks passed. This does not prove that the index belongs to this voice.']
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('The index changed during inspection. Wait for saving to finish and inspect again.')
    except Exception as error:
        details['inspection_error'] = str(error)[:1000]
    return details


def _index_summary(item):
    rows = ['\n### Index: ' + _md(Path(item['path']).name),
            '| Property | Information |', '| --- | --- |',
            '| Location | %s |' % _md(item['path']),
            '| Found by | %s |' % _md(item['match']),
            '| File size | %.2f MiB (%s bytes) |' % (item['size_bytes'] / 1024**2, format(item['size_bytes'], ','))]
    fields = [('last_modified_pc_local', 'Last file modification (PC local time)'),
              ('faiss_type', 'FAISS index type'), ('feature_dimension', 'Feature dimension'),
              ('rvc_version_suggested_by_dimension', 'RVC version suggested by dimension'),
              ('indexed_vectors', 'Stored feature vectors (not audio files)'),
              ('is_trained', 'Index training completed'), ('distance_metric', 'Distance metric'),
              ('cluster_count_nlist', 'IVF clusters (nlist)'),
              ('saved_search_clusters_nprobe', 'Saved search clusters (nprobe)'),
              ('code_bytes_per_vector', 'Encoded bytes per feature vector')]
    for key, label in fields:
        if key in item:
            value = item[key]
            if isinstance(value, bool): value = 'Yes' if value else 'No'
            rows.append('| %s | %s |' % (label, _md(value)))
    compatible = item.get('dimension_matches_model')
    rows.append('| Feature dimension matches model | %s |' % ('Yes; voice identity is not verified' if compatible is True else 'NO - incompatible dimension' if compatible is False else 'Unknown'))
    if item.get('sha256'):
        rows.append('\n**Index SHA-256 fingerprint:** `' + item['sha256'] + '`')
    if item.get('inspection_error'):
        rows.append('\n**Index could not be read:** ' + _md(item['inspection_error']))
    rows.extend('\n- ' + _md(note) for note in item.get('checks', []))
    return rows


def inspect_model(name, root):
    """Yield (percentage status, summary Markdown, metadata JSON, tensor Markdown).

    Only weights_only CPU loading is allowed. Missing metadata stays unknown;
    this inspection does not infer voice quality, training data or ownership.
    """
    progress = 0.0
    try:
        yield ('0.0% | Checking selected model...', '', {}, '')
        path = _path(name, root)
        before = path.stat()
        if not before.st_size:
            raise ValueError('The selected file is empty.')
        digest = hashlib.sha256()
        read_bytes = 0
        last_percent = -1
        with path.open('rb') as handle:
            while True:
                block = handle.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                read_bytes += len(block)
                percent = min(30, int(5 + 25 * read_bytes / before.st_size))
                if percent != last_percent:
                    progress = float(percent)
                    yield ('%.1f%% | Calculating file fingerprint: %.1f / %.1f MiB...'
                           % (progress, read_bytes / 1024**2, before.st_size / 1024**2), '', {}, '')
                    last_percent = percent
        progress = 35.0
        yield ('35.0% | Reading checkpoint on CPU...', '', {}, '')
        import torch
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('The checkpoint is not a supported RVC metadata dictionary.')
        weights = checkpoint.get('weight')
        if not isinstance(weights, dict) or not weights:
            if isinstance(checkpoint.get('model'), dict):
                raise ValueError('This is a training checkpoint, not an exported inference voice model (.pth with a weight dictionary).')
            raise ValueError('No RVC inference weight dictionary was found in this file.')
        progress = 55.0
        yield ('55.0% | Reading architecture and stored tensor information...', '', {}, '')
        config = checkpoint.get('config')
        config_valid = isinstance(config, (list, tuple)) and len(config) == 18
        config_data = dict(zip(CONFIG_FIELDS, _safe(config))) if config_valid else {}
        warnings = []
        if not config_valid:
            warnings.append('The standard 18-field RVC configuration is missing or unsupported.')
        rate_stored = _rate(checkpoint.get('sr'))
        rate_config = _rate(config[-1]) if config_valid else None
        if rate_stored and rate_config and rate_stored != rate_config:
            warnings.append('The stored sample rate and configuration sample rate disagree.')
        dtype_counts, dtype_elements = Counter(), Counter()
        elements = storage_bytes = 0
        tensor_rows = ['| Stored tensor | Shape | Type | Elements | Bytes |',
                       '| --- | --- | --- | ---: | ---: |']
        for i, (key, tensor) in enumerate(weights.items(), 1):
            if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError('The weight dictionary contains unsupported non-tensor values.')
            if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
                raise ValueError('The checkpoint contains unsupported tensor storage.')
            count = tensor.numel()
            size = count * tensor.element_size()
            dtype = str(tensor.dtype).replace('torch.', '')
            dtype_counts[dtype] += 1
            dtype_elements[dtype] += count
            elements += count
            storage_bytes += size
            if i <= 2500:
                tensor_rows.append('| %s | %s | %s | %s | %s |' % (
                    _md(key), _md(list(tensor.shape)), dtype, format(count, ','), format(size, ',')))
        if len(weights) > 2500:
            tensor_rows.append('\nOnly the first 2,500 tensor rows are displayed; totals include every tensor.')
        emb = weights.get('emb_g.weight')
        speakers = int(emb.shape[0]) if isinstance(emb, torch.Tensor) and emb.ndim == 2 else None
        emb_channels = int(emb.shape[1]) if isinstance(emb, torch.Tensor) and emb.ndim == 2 else None
        phone = weights.get('enc_p.emb_phone.weight')
        feature_dimension = int(phone.shape[1]) if isinstance(phone, torch.Tensor) and phone.ndim == 2 else None
        version = checkpoint.get('version', UNKNOWN)
        inferred_version = {256: 'v1', 768: 'v2'}.get(feature_dimension)
        if version in ('v1', 'v2') and inferred_version and version != inferred_version:
            warnings.append('Stored RVC version disagrees with the content projection input dimension.')
        if config_valid and speakers is not None and config[-3] != speakers:
            warnings.append('Configured speaker slots differ from the stored speaker embedding. RVC inference normally uses the embedding shape.')
        epoch, epoch_key = _first(checkpoint, ('epoch', 'epochs'))
        if epoch_key is None and isinstance(checkpoint.get('info'), str):
            match = re.fullmatch(r'\s*(\d+)\s*epochs?\s*', checkpoint['info'], flags=re.I)
            if match:
                epoch, epoch_key = int(match.group(1)), 'info field'
        steps, step_key = _first(checkpoint, ('step', 'steps', 'global_step', 'total_steps'))
        progress = 80.0
        yield ('80.0% | Looking for matching index files...', '', {}, '')
        indexes = _indexes(path, root)
        expected_dimension = feature_dimension or ({'v1': 256, 'v2': 768}.get(version) if isinstance(version, str) else None)
        for number, item in enumerate(indexes):
            progress = 80.0 + 15.0 * number / max(1, len(indexes))
            yield ('%.1f%% | Reading index %s/%s: %s...' % (progress, number + 1, len(indexes), Path(item['path']).name), '', {}, '')
            indexes[number] = _index_details(item, root, expected_dimension)
        mtime = datetime.fromtimestamp(before.st_mtime).strftime('%d-%m-%Y %H:%M:%S')
        metadata = {
            'file': {'name': path.name, 'relative_path': path.relative_to(Path(root).resolve()).as_posix(),
                     'size_bytes': before.st_size, 'size_mib': round(before.st_size / 1024**2, 3),
                     'last_modified_pc_local': mtime, 'sha256': digest.hexdigest()},
            'stored_metadata': {str(k)[:200]: _safe(v) for k, v in list(checkpoint.items())[:100]
                                if k not in ('weight', 'config') and isinstance(k, (str, int))},
            'architecture': config_data or {'raw_config': _safe(config)},
            'observed_weights': {'tensor_count': len(weights), 'stored_elements': elements,
                                 'tensor_bytes': storage_bytes, 'tensor_counts_by_dtype': dict(dtype_counts),
                                 'elements_by_dtype': dict(dtype_elements), 'speaker_slots': speakers,
                                 'speaker_embedding_channels': emb_channels,
                                 'content_projection_input_dimension': feature_dimension,
                                 'version_suggested_by_projection_dimension': inferred_version},
            'matching_indexes': indexes, 'warnings': warnings,
        }
        f0 = checkpoint.get('f0', UNKNOWN)
        f0_text = 'Yes' if f0 in (1, True) else ('No' if f0 in (0, False) else UNKNOWN)
        rate_text = ('%s Hz (%s kHz)' % (rate_config or rate_stored, (rate_config or rate_stored) / 1000)) if rate_config or rate_stored else UNKNOWN
        summary = [
            '### ' + _md(path.name), '| Property | Information |', '| --- | --- |',
            '| File size | %.2f MiB (%s bytes) |' % (before.st_size / 1024**2, format(before.st_size, ',')),
            '| Last file modification (PC local time) | %s |' % mtime,
            '| Sample rate | %s |' % rate_text,
            '| Stored RVC version | %s |' % _md(version),
            '| Pitch guidance (stored F0 flag) | %s |' % f0_text,
            '| Vocoder | %s |' % _md(checkpoint.get('vocoder', 'HiFi-GAN')),
            '| Embedder Model | %s |' % _md(checkpoint.get('embedder_model', 'contentvec')),
            '| Speaker slots in the stored embedding | %s |' % (speakers if speakers is not None else UNKNOWN),
            '| Speaker embedding channels | %s |' % (emb_channels if emb_channels is not None else UNKNOWN),
            '| Content projection input dimension | %s |' % (feature_dimension if feature_dimension is not None else UNKNOWN),
            '| Saved epoch information | %s%s |' % (_md(epoch), ' (from %s)' % epoch_key if epoch_key else ''),
            '| Saved training steps | %s%s |' % (_md(steps), ' (from %s)' % step_key if step_key else ''),
            '| Stored tensors | %s |' % format(len(weights), ','),
            '| Stored weight elements | %s |' % format(elements, ','),
            '| Tensor data size | %.2f MiB |' % (storage_bytes / 1024**2),
            '| Stored tensor types | %s |' % ', '.join('%s: %s tensors' % (key, value) for key, value in sorted(dtype_counts.items())),
            '\n**SHA-256 file fingerprint:** `' + digest.hexdigest() + '`',
        ]
        if checkpoint.get('info') is not None:
            summary.extend(['\n**Saved model notes:**', _md(_safe(checkpoint['info']))])
        if warnings:
            summary.extend(['\n**Metadata checks:**'] + ['- ' + _md(warning) for warning in warnings])
        summary.append('\n**Matching index files:**')
        summary.extend(['- %s (%.2f MiB; %s)' % (_md(item['path']), item['size_bytes'] / 1024**2, item['match']) for item in indexes]
                       or ['No matching index was found by folder or filename.'])
        for item in indexes:
            summary.extend(_index_summary(item))
        if indexes:
            summary.append('\nIndex vectors are extracted audio features, not a recording count or dataset duration. nlist is the number of IVF groups; nprobe is the saved search setting and may be changed by the inference program. Matching dimensions are necessary, but do not prove that an index belongs to a particular voice. Index files do not reliably store sample rate, training epochs or audio quality. Unreadable indexes are reported separately; model information remains available.')
        summary.extend(['\n**What this file cannot tell you:**',
                        'Dataset length, audio quality, consent, source speaker, training GPU, batch size, exact embedder, pretrained model and total training time are usually not stored in an inference export. They are not guessed here. Any explicitly saved extra metadata appears below.',
                        'Speaker slots are embedding entries, not proof that every slot represents a trained voice. File modification time is not the training completion date. The tensor data size is not a prediction of inference or training VRAM. Index matches are based on names, not a content-compatibility test.'])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('The file changed during inspection. Wait for saving to finish and inspect it again.')
        yield ('100.0% | Model information ready.', '\n'.join(summary), metadata, '\n'.join(tensor_rows))
    except Exception as error:
        yield ('%.1f%% | Inspection failed: %s' % (progress, str(error)[:1200]), '', {}, '')
