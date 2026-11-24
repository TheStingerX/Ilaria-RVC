"""RVC decoder selection shared by training, inference and model import checks."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
VOCODERS = ('HiFi-GAN', 'MRF HiFi-GAN', 'RefineGAN')
_MEL_LOSSES = {}


def normalize_vocoder(value=None):
    value = value or 'HiFi-GAN'
    if value not in VOCODERS:
        raise ValueError('Choose HiFi-GAN, MRF HiFi-GAN, or RefineGAN.')
    return value


def apply_vocoder(model, vocoder='HiFi-GAN', version='v2', if_f0=1, sample_rate=32000):
    """Replace the waveform decoder before loading any weights or optimizer."""
    vocoder = normalize_vocoder(vocoder)
    if vocoder == 'HiFi-GAN':
        model.vocoder = vocoder
        return model
    sample_rate = {'32k': 32000, '40k': 40000, '44.1k': 44100, '48k': 48000}.get(sample_rate, sample_rate)
    if vocoder == 'MRF HiFi-GAN':
        if version != 'v2' or not if_f0 or int(sample_rate) not in (32000, 40000, 48000):
            raise ValueError('MRF HiFi-GAN requires RVC v2, 32/40/48 kHz, and Pitch Guidance.')
        from ilaria_hifigan_mrf import HiFiGANMRFGenerator
        model.dec = HiFiGANMRFGenerator(
            in_channel=model.inter_channels,
            upsample_initial_channel=model.upsample_initial_channel,
            upsample_rates=list(model.upsample_rates),
            upsample_kernel_sizes=list(model.upsample_kernel_sizes),
            resblock_kernel_sizes=model.resblock_kernel_sizes,
            resblock_dilations=model.resblock_dilation_sizes,
            gin_channels=model.gin_channels,
            sample_rate=int(sample_rate),
            harmonic_num=8,
        )
        model.vocoder = vocoder
        return model
    if version != 'v2' or not if_f0 or int(sample_rate) not in (32000, 40000, 44100):
        raise ValueError('RefineGAN requires RVC v2, 32, 40 or 44.1 kHz and Pitch Guidance enabled.')
    expected_rates = {32000: [10, 8, 2, 2], 40000: [10, 10, 2, 2], 44100: [7, 7, 3, 3]}
    if list(model.upsample_rates) != expected_rates[int(sample_rate)] or int(model.gin_channels) != 256:
        raise ValueError('This RefineGAN port requires the standard v2 architecture for the selected sample rate.')
    from ilaria_refinegan import RefineGANGenerator
    model.dec = RefineGANGenerator(sample_rate=int(sample_rate),
                                   upsample_rates=list(model.upsample_rates),
                                   downsample_rates=list(model.upsample_rates)[::-1],
                                   num_mels=model.inter_channels,
                                   gin_channels=model.gin_channels,
                                   start_channels=16,
                                   upsample_initial_channel=512)
    model.vocoder = vocoder
    return model



def refinegan_44100_config(base):
    from copy import deepcopy

    config = deepcopy(base)
    config["data"].update(sampling_rate=44100, hop_length=441)
    config["train"]["segment_size"] = 441 * 32
    config["model"].update(upsample_rates=[7, 7, 3, 3], upsample_kernel_sizes=[15, 15, 7, 7])
    return config


def prepare_refinegan_mute(root):
    """Create the 44.1 kHz mute waveform from the bundled 40 kHz sample."""
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    folder = Path(root) / "logs" / "mute" / "0_gt_wavs"
    target = folder / "mute44.1k.wav"
    if target.is_file():
        return target
    rate, audio = wavfile.read(folder / "mute40k.wav")
    if rate != 40000:
        raise ValueError("Expected the bundled 40 kHz mute waveform.")
    result = resample_poly(audio.astype(np.float32), 441, 400)
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        result = np.clip(result.round(), limits.min, limits.max)
    with NamedTemporaryFile(dir=folder, suffix=".wav", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        wavfile.write(temporary, 44100, result.astype(audio.dtype))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def weights_for_runtime(weights):
    """Translate newer Applio parametrization names to legacy PyTorch 2.0 names."""
    result = {}
    for key, value in weights.items():
        key = key.replace('.parametrizations.weight.original0', '.weight_g')
        key = key.replace('.parametrizations.weight.original1', '.weight_v')
        if key in result:
            raise ValueError('Conflicting weight-normalization keys in this checkpoint.')
        result[key] = value
    return result


def inferred_vocoder(weights):
    """Recognize decoder architecture independently of optional metadata."""
    names = tuple(weights)
    if any(name.startswith('dec.mel_conv.') for name in names):
        return 'RefineGAN'
    if any(name.startswith('dec.mrfs.') for name in names):
        return 'MRF HiFi-GAN'
    if any(name.startswith('dec.conv_pre.') for name in names):
        return 'HiFi-GAN'
    return None


def checkpoint_vocoder(checkpoint):
    weights = checkpoint.get('weight', checkpoint.get('model', {}))
    observed = inferred_vocoder(weights)
    stored = checkpoint.get('vocoder')
    if stored is not None:
        stored = normalize_vocoder(stored)
        if observed and observed != stored:
            raise ValueError('Saved vocoder metadata disagrees with decoder weights.')
    return stored or observed or 'HiFi-GAN'


def validate_pretrain(checkpoint, vocoder):
    observed = inferred_vocoder(checkpoint.get('model', checkpoint.get('weight', {})))
    if observed is not None and observed != normalize_vocoder(vocoder):
        raise ValueError('The generator pretrain uses %s, but %s is selected.' % (observed, vocoder))


def architecture_metadata(hps):
    train = getattr(hps, 'train', hps)
    return {'vocoder': normalize_vocoder(getattr(train, 'vocoder', getattr(hps, 'vocoder', 'HiFi-GAN'))),
            'embedder_model': getattr(train, 'embedder_model', getattr(hps, 'embedder_model', 'contentvec'))}


def build_refinegan_discriminator(use_spectral_norm=False):
    from ilaria_refinegan_discriminator import MultiPeriodDiscriminator
    return MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm, version='v3')


def refinegan_mel_loss(real, fake, sample_rate=32000):
    """Applio's multi-scale waveform mel loss; caller applies c_mel / 3."""
    key = int(sample_rate)
    if key not in _MEL_LOSSES:
        from ilaria_refinegan_loss import MultiScaleMelSpectrogramLoss
        _MEL_LOSSES[key] = MultiScaleMelSpectrogramLoss(sample_rate=key)
    return _MEL_LOSSES[key](real.float(), fake.float())
