"""Discover installed generator/discriminator pairs without loading neural weights."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path
import re

AUTO = 'Auto (official pretrained for current settings)'
NONE = 'None (train from scratch)'
CUSTOM = 'Custom paths'
SELECT = 'Select a pretrained pair'
FROZEN = 'Saved pretrained pair'
EXCLUDED = {'weights', 'uvr5_weights', 'uvr', 'hubert', 'rmvpe', 'training-state', 'logs', '.git'}
PAIR_STEM = re.compile(r'^(f0)?([gd])(.*)$', re.I)


def _pair_role(stem):
    named = re.fullmatch(r'(generator|discriminator)(?:[_ .-]*(.*))?', stem, re.I)
    if named:
        return named[1][0].upper(), False, (named[2] or 'model').casefold()
    suffix = re.fullmatch(r'(.+?)[_ .-]+(g|d|generator|discriminator)', stem, re.I)
    if suffix:
        return suffix[2][0].upper(), False, suffix[1].casefold()
    if re.match(r'^(?:f0)?[gd](?:[_-]|[0-9]|$)', stem, re.I):
        match = PAIR_STEM.fullmatch(stem)
        return match[2].upper(), bool(match[1]), (match[3].strip('_ .-') or 'model').casefold()
    middle = re.fullmatch(r'(.+?)[_-]([gGdD])[_-](.+)', stem)
    if middle:
        return middle[2].upper(), False, (middle[1]+'_'+middle[3]).casefold()
    attached = re.fullmatch(r'(.+?)([GD])(_.+)', stem)
    if attached:
        return attached[2], False, (attached[1]+attached[3]).casefold()
    attached_end = re.fullmatch(r'(.+?)([GD])', stem)
    if attached_end:
        return attached_end[2], stem.lower().startswith('f0'), attached_end[1].casefold()
    match = PAIR_STEM.fullmatch(stem)
    if match:
        return match[2].upper(), bool(match[1]), (match[3].strip('_ .-') or 'model').casefold()
    return None


def _stock_pair(generator):
    path = Path(generator)
    match = re.fullmatch(r'(f0)?G(32|40|48)k', path.stem, re.I)
    if match and path.parent.name in ('pretrained', 'pretrained_v2'):
        return ('v2' if path.parent.name == 'pretrained_v2' else 'v1', match[2]+'k', bool(match[1]))
    return None


def _pairs(root):
    root = Path(root).resolve()
    assets = root / 'assets'
    if not assets.is_dir():
        return {}
    candidates = {}
    for path in assets.rglob('*.pth'):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(assets.resolve())
        except ValueError:
            continue
        if any(part.casefold() in EXCLUDED for part in relative.parts[:-1]):
            continue
        role = _pair_role(path.stem)
        if role is None:
            continue
        letter, f0, stem = role
        key = (str(resolved.parent).casefold(), f0, stem)
        candidates.setdefault(key, {})[letter] = resolved
    pairs = {}
    for pair in sorted(candidates.values(), key=lambda item: str(item.get("G", "")).casefold()):
        if 'G' not in pair or 'D' not in pair:
            continue
        generator, discriminator = pair['G'], pair['D']
        folder = generator.parent.relative_to(root).as_posix()
        stock = _stock_pair(generator)
        if stock:
            version, rate, pitch = stock
            label = 'Original RVC %s | %s | %s' % (version, rate, 'Pitch guidance' if pitch else 'No pitch guidance')
        else:
            label = '%s + %s' % (generator.name, discriminator.name)
            base_label, number = label, 2
            while label in pairs:
                label = '%s (%s)' % (base_label, number)
                number += 1
        pairs[label] = (generator.relative_to(root).as_posix(), discriminator.relative_to(root).as_posix())
    return dict(sorted(pairs.items(), key=lambda item: item[0].casefold()))


def pretrain_choices(root, rate=None, pitch_guidance=None, version=None, vocoder=None):
    choices = [SELECT, NONE] if vocoder == 'RefineGAN' else [AUTO, NONE]
    for label, pair in _pairs(root).items():
        stock = _stock_pair(pair[0])
        if stock:
            if vocoder == 'RefineGAN':
                continue
            if version is not None and stock[0] != version:
                continue
            if rate is not None and stock[1] != rate:
                continue
            if pitch_guidance is not None and stock[2] != bool(pitch_guidance):
                continue
        choices.append(label)
    return choices


def _pitch_enabled(value):
    if isinstance(value, bool) or isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in ('true', 'false', 'yes', 'no', '1', '0'):
        return value.strip().casefold() in ('true', 'yes', '1')
    raise ValueError('Pitch Guidance must be True or False.')


def apply_pretrain(choice, rate, pitch_guidance, version, root):
    """Return (G path, D path, status); None paths mean keep custom textboxes.

    Auto missing files are returned as their expected paths, with an explicit
    message, so launch-time path validation can block accidental scratch training.
    This function never downloads files, reads checkpoints or guesses architecture.
    """
    if choice == CUSTOM:
        return None, None, 'Custom paths: keep or edit both pretrained paths below. Their architecture must match the training settings.'
    if choice == NONE:
        return '', '', 'No pretrained pair: this model will train from scratch.'
    root = Path(root).resolve()
    if choice == AUTO:
        if version not in ('v1', 'v2') or rate not in ('32k', '40k', '48k'):
            raise ValueError('Choose a supported RVC version and sample rate first.')
        folder = 'assets/pretrained_v2' if version == 'v2' else 'assets/pretrained'
        prefix = 'f0' if _pitch_enabled(pitch_guidance) else ''
        generator, discriminator = ('%s/%s%s%s.pth' % (folder, prefix, letter, rate) for letter in ('G','D'))
        if not (root / generator).is_file() or not (root / discriminator).is_file():
            return generator, discriminator, 'Official pretrained pair is not installed for these settings. Install both files or select another compatible pair; training must not silently start from scratch.'
        return generator, discriminator, 'Official pair selected for %s, %s, Pitch Guidance %s.' % (version, rate, prefix == 'f0')
    pairs = _pairs(root)
    pair = pairs.get(choice)
    if pair is None:
        # Preserve saved queue jobs using the former path-based labels.
        pair = next((value for value in pairs.values() if '%s | %s + %s' % (
            Path(value[0]).parent.as_posix(), Path(value[0]).name, Path(value[1]).name) == choice), None)
    if not pair:
        raise ValueError('The selected pretrained pair is unavailable. Refresh the pretrained list.')
    generator, discriminator = pair
    status = 'Installed pair selected. Its sample rate, pitch guidance, RVC version and vocoder must match this training.'
    named_rate = re.search(r'(?:32|40|48)k', Path(generator).stem, re.I)
    if named_rate and named_rate.group(0).lower() != rate:
        status = 'Warning: the selected filename suggests %s, but Sample Rate is %s. Choose compatible training settings.' % (named_rate.group(0).lower(), rate)
    return generator, discriminator, status


def validate_pretrain_paths(generator, discriminator, root):
    """Reject missing/half-pairs before launching; return canonical local paths.

    Architecture compatibility remains the trainer's strict state-dict/shape check.
    Loading huge weights twice merely to populate a dropdown would waste memory.
    """
    generator, discriminator = str(generator or '').strip(), str(discriminator or '').strip()
    if not generator and not discriminator:
        return '', ''
    if not generator or not discriminator:
        raise ValueError('Select both pretrained files (G and D), or choose None to train from scratch.')
    paths = []
    for label, value in [('Generator', generator), ('Discriminator', discriminator)]:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve()
        if path.suffix.lower() != '.pth' or not path.is_file() or path.stat().st_size == 0:
            raise ValueError('%s pretrained file is missing or empty. Select an installed compatible pair.' % label)
        paths.append(str(path))
    return tuple(paths)


def resolve_training_pretrain(choice, rate, pitch, version, root, generator='', discriminator=''):
    """Resolve both paths together. Saved queue jobs keep their captured pair."""
    if choice not in (CUSTOM, FROZEN):
        generator, discriminator, _ = apply_pretrain(choice, rate, pitch, version, root)
    elif not generator or not discriminator:
        raise ValueError('The saved pretrained pair is incomplete. Select a complete installed pair.')
    return validate_pretrain_paths(generator, discriminator, root)
