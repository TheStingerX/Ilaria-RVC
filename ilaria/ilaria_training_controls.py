"""Keep training controls consistent with the selected backend capabilities."""
from ilaria_pretrain_selector import AUTO, NONE, SELECT, apply_pretrain, pretrain_choices


def control_updates(root, version, vocoder, rate, pitch, embedder, target, cache,
                    precision, pretrain, generator, discriminator, method, preview,
                    cuda=True, no_value='No', gpu_pitch=True):
    def update(current=None, value=None, **properties):
        if value is not None and current != value:
            properties['value'] = value
        return dict(properties, __type__='update')

    original_vocoder, original_embedder = vocoder, embedder
    if version == 'v1':
        vocoder, embedder = 'HiFi-GAN', 'contentvec'
    refined = vocoder == 'RefineGAN'
    rates = ['32k'] if refined else (['40k', '48k'] if version == 'v1' else ['32k', '40k', '48k'])
    chosen_rate = rate if rate in rates else rates[0]
    chosen_pitch = True if refined else bool(pitch)
    vocoders = ['HiFi-GAN', 'RefineGAN'] if version == 'v2' and chosen_pitch else ['HiFi-GAN']
    embedders = ['contentvec', 'spin-v2'] if version == 'v2' else ['contentvec']
    choices = pretrain_choices(root, chosen_rate, chosen_pitch, version, vocoder)
    selected = pretrain if pretrain in choices else (SELECT if refined else AUTO)
    try:
        if selected == SELECT:
            g, d = '', ''
            message = 'Select an installed compatible pretrained pair, or choose None to train from scratch. Download pretrained pairs in Other, then refresh this list.'
        else:
            g, d, message = apply_pretrain(selected, chosen_rate, chosen_pitch, version, root)
        if g and d:
            message += ' G: %s | D: %s' % (g, d)
    except (OSError, ValueError) as error:
        g, d, message = '', '', str(error)
    methods = ['rmvpe', 'swift', 'beatrice', 'fcpe', 'parselmouth', 'harvest', 'dio']
    if cuda and gpu_pitch:
        methods.append('rmvpe_gpu')
    chosen_method = method if method in methods else ('parselmouth' if method == 'pm' else 'rmvpe')
    cache_enabled = cuda and target != 'Steps'
    selected_cache = cache if cache_enabled else no_value
    selected_precision = precision if cuda else 'FP32'
    return (
        update(rate, chosen_rate, choices=rates, interactive=len(rates)>1),
        update(pitch, chosen_pitch, choices=[True] if refined else [True,False], interactive=not refined),
        update(original_vocoder, vocoder, choices=vocoders, interactive=len(vocoders)>1),
        update(original_embedder, embedder, choices=embedders, interactive=len(embedders)>1),
        update(cache, selected_cache, interactive=cache_enabled),
        update(precision, selected_precision, choices=['FP16','FP32'] if cuda else ['FP32'], interactive=cuda),
        update(pretrain, selected, choices=choices),
        g,
        d,
        message,
        update(method, chosen_method, choices=methods, visible=chosen_pitch, interactive=chosen_pitch),
        update(visible=chosen_pitch and chosen_method=='rmvpe_gpu' and cuda and gpu_pitch, interactive=cuda and gpu_pitch),
        update(visible=target=='Epochs'),
        update(visible=target=='Steps'),
        update(visible=bool(preview), interactive=bool(preview)),
        update(visible=bool(preview), interactive=bool(preview)),
        update(visible=bool(preview), interactive=bool(preview)),
        update(visible=bool(preview), interactive=bool(preview)),
        update(interactive=cuda),
        update(interactive=cuda),
    )
