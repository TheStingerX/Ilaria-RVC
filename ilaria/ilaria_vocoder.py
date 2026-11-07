"""RVC decoder selection shared by training, inference and model import checks."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
VOCODERS = ('HiFi-GAN', 'RefineGAN')
_MEL_LOSSES = {}


def normalize_vocoder(value=None):
    value = value or 'HiFi-GAN'
    if value not in VOCODERS:
        raise ValueError('Choose HiFi-GAN or RefineGAN.')
    return value


def apply_vocoder(model, vocoder='HiFi-GAN', version='v2', if_f0=1, sample_rate=32000):
    """Replace the waveform decoder before loading any weights or optimizer."""
    vocoder = normalize_vocoder(vocoder)
    if vocoder == 'HiFi-GAN':
        model.vocoder = vocoder
        return model
    sample_rate = {'32k': 32000, '40k': 40000, '48k': 48000}.get(sample_rate, sample_rate)
    if version != 'v2' or not if_f0 or int(sample_rate) != 32000:
        raise ValueError('RefineGAN requires RVC v2, 32 kHz and Pitch Guidance enabled.')
    if list(model.upsample_rates) != [10, 8, 2, 2] or int(model.gin_channels) != 256:
        raise ValueError('This RefineGAN port requires the standard v2 32 kHz architecture.')
    from ilaria_refinegan import RefineGANGenerator
    model.dec = RefineGANGenerator(sample_rate=32000,
                                   upsample_rates=list(model.upsample_rates),
                                   downsample_rates=list(model.upsample_rates)[::-1],
                                   num_mels=model.inter_channels,
                                   gin_channels=model.gin_channels,
                                   start_channels=16,
                                   upsample_initial_channel=512)
    model.vocoder = vocoder
    return model


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
