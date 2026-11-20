"""User-selected pretrained catalog; source filenames verified on 2026-11-05."""
from urllib.parse import quote
CATALOG = ({'name': 'DMR v2', 'repo': 'ourbz/DMR-V2-ReUpload', 'g': 'G_DMR-V2.pth', 'd': 'D_DMR-V2.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'DMR v2 | 32 kHz | HiFi-GAN'}, {'name': 'GuideVocalPretrain', 'repo': 'Essid/GuideVocalPretrain', 'g': 'G_GuideVocalPretrain.pth', 'd': 'D_GuideVocalPretrain.pth', 'sample_rate': 48000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'GuideVocalPretrain | 48 kHz | HiFi-GAN'}, {'name': 'IMA', 'repo': 'Loren85/IMA-TEST-V1', 'g': 'G_2333333.pth', 'd': 'D_2333333.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'IMA | 32 kHz | HiFi-GAN'}, {'name': 'Itaila', 'repo': 'TheStinger/itaila', 'g': 'ItaIla_32k_G.pth', 'd': 'ItaIla_32k_D.pth', 'sample_rate': 32000, 'vocoder': 'Not specified', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'Itaila | 32 kHz'}, {'name': 'KLM - BeatzForge V2', 'repo': 'SeoulStreamingStation/KLM_BEATMASTER', 'g': 'G_BeatzForge_V2_32k.pth', 'd': 'D_BeatzForge_V2_32k.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM - BeatzForge V2 | 32 kHz | HiFi-GAN'}, {'name': 'KLM 4 HiFi-GAN', 'repo': 'SeoulStreamingStation/KLM49_HFG', 'g': 'G_KLM_HFG_32k.pth', 'd': 'D_KLM_HFG_32k.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM 4 HiFi-GAN | 32 kHz | HiFi-GAN'}, {'name': 'KLM 4 HiFi-GAN', 'repo': 'SeoulStreamingStation/KLM49_HFG', 'g': 'G_KLM_HFG_40k.pth', 'd': 'D_KLM_HFG_40k.pth', 'sample_rate': 40000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM 4 HiFi-GAN | 40 kHz | HiFi-GAN'}, {'name': 'KLM 4 HiFi-GAN', 'repo': 'SeoulStreamingStation/KLM49_HFG', 'g': 'G_KLM_HFG_48k.pth', 'd': 'D_KLM_HFG_48k.pth', 'sample_rate': 48000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM 4 HiFi-GAN | 48 kHz | HiFi-GAN'}, {'name': 'KLM 5 RefineGAN', 'repo': 'SeoulStreamingStation/KLM5', 'g': 'G_KLM50_RFG_32Khz.pth', 'd': 'D_KLM50_RFG_32Khz.pth', 'sample_rate': 32000, 'vocoder': 'RefineGAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM 5 RefineGAN | 32 kHz | RefineGAN'}, {'name': 'KLM 5 RefineGAN', 'repo': 'SeoulStreamingStation/KLM5', 'g': 'G_KLM50_RFG_40Khz.pth', 'd': 'D_KLM50_RFG_40Khz.pth', 'sample_rate': 40000, 'vocoder': 'RefineGAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'KLM 5 RefineGAN | 40 kHz | RefineGAN'}, {'name': 'KLM 6.1 (Experimental V3 L608)', 'repo': 'SeoulStreamingStation/KLM6_Experimental', 'g': 'G_KLM6_Exp3_L6_32k.pth', 'd': 'D_KLM6_Exp3_L6_32k.pth', 'sample_rate': 32000, 'vocoder': 'Not specified', 'embedder': 'Not specified', 'variant': 'Linked Exp3 L6 files', 'note': 'The supplied description also calls this KLM 6.2; the exact linked Exp3_L6 pair is used.', 'label': 'KLM 6.1 (Experimental V3 L608) | Linked Exp3 L6 files | 32 kHz'}, {'name': 'Mandagues V2', 'repo': 'juaow/mandaguesv2', 'g': 'mandaguesG_40k.pth', 'd': 'mandaguesD_40k.pth', 'sample_rate': 40000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'Mandagues V2 | 40 kHz | HiFi-GAN'}, {'name': 'Nanashi Anime', 'repo': 'shiromiya/nanashi-pretrain', 'g': 'v1_anime/normal/G_nanashi_anime_384e.pth', 'd': 'v1_anime/normal/D_nanashi_anime_384e.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Normal', 'note': '', 'label': 'Nanashi Anime | Normal | 32 kHz | HiFi-GAN'}, {'name': 'Nanashi v1', 'repo': 'shiromiya/nanashi-pretrain', 'g': 'v1.7/G_nanashi_v1_7.pth', 'd': 'v1.7/D_nanashi_v1_7.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'v1.7', 'note': '', 'label': 'Nanashi v1 | v1.7 | 32 kHz | HiFi-GAN'}, {'name': 'Nanashi v2 Beta', 'repo': 'shiromiya/nanashi-pretrain', 'g': 'v2/base/G_nanashi_v2_800e_1005600s.pth', 'd': 'v2/base/D_nanashi_v2_800e_1005600s.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Base', 'note': '', 'label': 'Nanashi v2 Beta | Base | 32 kHz | HiFi-GAN'}, {'name': 'Nanashi v2 Beta', 'repo': 'shiromiya/nanashi-pretrain', 'g': 'v2/finetune/G_nanashi_v2_ft_270e_143910s.pth', 'd': 'v2/finetune/D_nanashi_v2_ft_270e_143910s.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Fine-tuned', 'note': '', 'label': 'Nanashi v2 Beta | Fine-tuned | 32 kHz | HiFi-GAN'}, {'name': 'Ov2Super', 'repo': 'poiqazwsx/Ov2Super32kfix', 'g': 'f0Ov2Super32kG.pth', 'd': 'f0Ov2Super32kD.pth', 'sample_rate': 32000, 'vocoder': 'Not specified', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'Ov2Super | 32 kHz'}, {'name': 'PaBP', 'repo': 'Razer112/pabp-v.5', 'g': 'G_31132.pth', 'd': 'D_31132.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'PaBP | 32 kHz | HiFi-GAN'}, {'name': 'RefGanVCTKV2', 'repo': 'SimplCup/RefGanVCTKV2', 'g': 'f0G32k.pth', 'd': 'f0D32k.pth', 'sample_rate': 32000, 'vocoder': 'RefineGAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'RefGanVCTKV2 | 32 kHz | RefineGAN'}, {'name': 'RefGanVCTKV2', 'repo': 'SimplCup/RefGanVCTKV2', 'g': 'f0G40k.pth', 'd': 'f0D40k.pth', 'sample_rate': 40000, 'vocoder': 'RefineGAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'RefGanVCTKV2 | 40 kHz | RefineGAN'}, {'name': 'RefineGAN Base Pretrains (Experimental)', 'repo': 'Aznamir/RefineGAN', 'g': 'RFGv3_CV_G_1771500.pth', 'd': 'RFGv3_CV_D_1771500.pth', 'sample_rate': 32000, 'vocoder': 'RefineGAN', 'embedder': 'contentvec', 'variant': 'v3', 'note': '', 'label': 'RefineGAN Base Pretrains (Experimental) | v3 | 32 kHz | RefineGAN | contentvec'}, {'name': 'RefineGAN Base Pretrains (Experimental)', 'repo': 'Aznamir/RefineGAN', 'g': 'RFGv3_SPv2_G_1771500.pth', 'd': 'RFGv3_SPv2_D_1771500.pth', 'sample_rate': 32000, 'vocoder': 'RefineGAN', 'embedder': 'spin-v2', 'variant': 'v3', 'note': '', 'label': 'RefineGAN Base Pretrains (Experimental) | v3 | 32 kHz | RefineGAN | spin-v2'}, {'name': 'RefineGanVCTKV1', 'repo': 'SimplCup/RefineGanVCTKV1', 'g': 'f0G_RefineGanVCTKV1.pth', 'd': 'f0D_RefineGanVCTKV1.pth', 'sample_rate': 44100, 'vocoder': 'RefineGAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'RefineGanVCTKV1 | 44.1 kHz | RefineGAN'}, {'name': 'Rigel', 'repo': 'MUSTAR/Rigel-rvc-base-pretrained-model', 'g': 'Rigel_32k_Base_and_FineTuned/Base-model_32k_fp32/G_Rigel_32k_3890220.pth', 'd': 'Rigel_32k_Base_and_FineTuned/Base-model_32k_fp32/D_Rigel_32k_3890220.pth', 'sample_rate': 32000, 'vocoder': 'Not specified', 'embedder': 'Not specified', 'variant': 'Base', 'note': '', 'label': 'Rigel | Base | 32 kHz'}, {'name': 'Rigel', 'repo': 'MUSTAR/Rigel-rvc-base-pretrained-model', 'g': 'Rigel_32k_Base_and_FineTuned/FineTuned-model_32k_fp32/G_Rigel_32k_fp32_2854856.pth', 'd': 'Rigel_32k_Base_and_FineTuned/FineTuned-model_32k_fp32/D_Rigel_32k_fp32_2854856.pth', 'sample_rate': 32000, 'vocoder': 'Not specified', 'embedder': 'Not specified', 'variant': 'Fine-tuned', 'note': '', 'label': 'Rigel | Fine-tuned | 32 kHz'}, {'name': 'SingerPreTrain', 'repo': 'Sztef/SingerPreTrained', 'g': 'update/f0G_SingerPreTrain.pth', 'd': 'update/f0D_SingerPreTrain.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Updated', 'note': '', 'label': 'SingerPreTrain | Updated | 32 kHz | HiFi-GAN'}, {'name': 'Titan', 'repo': 'blaise-tk/TITAN', 'g': 'models/medium/32k/pretrained/G-f032k-TITAN-Medium.pth', 'd': 'models/medium/32k/pretrained/D-f032k-TITAN-Medium.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Medium', 'note': '', 'label': 'Titan | Medium | 32 kHz | HiFi-GAN'}, {'name': 'Titan', 'repo': 'blaise-tk/TITAN', 'g': 'models/medium/40k/pretrained/G-f040k-TITAN-Medium.pth', 'd': 'models/medium/40k/pretrained/D-f040k-TITAN-Medium.pth', 'sample_rate': 40000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Medium', 'note': '', 'label': 'Titan | Medium | 40 kHz | HiFi-GAN'}, {'name': 'Titan', 'repo': 'blaise-tk/TITAN', 'g': 'models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth', 'd': 'models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth', 'sample_rate': 48000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': 'Medium', 'note': '', 'label': 'Titan | Medium | 48 kHz | HiFi-GAN'}, {'name': 'UKA', 'repo': 'Plasmati/Pretrains', 'g': 'UKA-Pretrain-G.pth', 'd': 'UKA-Pretrain-D.pth', 'sample_rate': 32000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified', 'variant': '', 'note': '', 'label': 'UKA | 32 kHz | HiFi-GAN'})


KLM_MRF_PRETRAINS = tuple(
    {
        'name': 'KLM 4.4',
        'repo': 'SeoulStreamingStation/KLM4.4',
        'g': 'G_KLM44_' + filename,
        'd': 'D_KLM44_' + filename,
        'sample_rate': 40000,
        'vocoder': 'MRF HiFi-GAN',
        'embedder': 'Not specified',
        'variant': variant,
        'note': 'Publisher provides an MRF G/D pair. Training compatibility has not been tested in Ilaria RVC.',
        'label': f'KLM 4.4 {variant} | 40 kHz | MRF HiFi-GAN',
    }
    for variant, filename in (('X1', 'X1_MRF_40khz.pth'), ('X2', 'x2_MRF_40k.pth'))
)
CATALOG = tuple(sorted((*CATALOG, *KLM_MRF_PRETRAINS), key=lambda entry: entry['label'].casefold()))


POLITREES_PRETRAINS = (
    ('AnimePretrain', 32, 'Anime', 'f0G_AnimePreTrain.pth', 'f0D_AnimePreTrain.pth'),
    ('Nezox', 32, 'Nezox', 'Nezox_32k_G.pth', 'Nezox_32k_D.pth'),
    ('UKR', 32, 'UK', 'UKR-Pretrain-G.pth', 'UKR-Pretrain-D.pth'),
    ('RIN_E3', 40, 'RIN_E', 'G_RIN_E3.pth', 'D_RIN_E3.pth'),
    ('Snowie', 40, 'Snowie', 'G_Snowie_40k.pth', 'D_Snowie_40k.pth'),
    ('Snowie V2', 40, 'Snowie', 'G_SnowieV2_40k.pth', 'D_SnowieV2_40k.pth'),
    ('Snowie V2', 48, 'Snowie', 'G_SnowieV2_48k.pth', 'D_SnowieV2_48k.pth'),
    ('Snowie V3.1', 32, 'Snowie', 'G_SnowieV3.1_32k.pth', 'D_SnowieV3.1_32k.pth'),
    ('Snowie V3.1', 40, 'Snowie', 'G_SnowieV3.1_40k.pth', 'D_SnowieV3.1_40k.pth'),
    ('Snowie V3.1', 48, 'Snowie', 'G_SnowieV3.1_48k.pth', 'D_SnowieV3.1_48k.pth'),
    ('Snowie × RIN', 40, 'Snowie', 'G_Snowie-X-Rin_40k.pth', 'D_Snowie-X-Rin_40k.pth'),
    ('VocalCore', 48, 'VocalCore', 'G_VocalCore.pth', 'D_VocalCore.pth'),
    ('VocalCore Try3b', 48, 'VocalCore', 'G_VocalCoreTry3b.pth', 'D_VocalCoreTry3b.pth'),
)


def _politrees_entry(name, rate, folder, generator, discriminator):
    prefix = f'pretrained/v2/{rate}k/{folder}/'
    return {
        'name': name, 'repo': 'Politrees/RVC_resources',
        'g': prefix + generator, 'd': prefix + discriminator,
        'sample_rate': rate * 1000, 'vocoder': 'HiFi-GAN', 'embedder': 'Not specified',
        'variant': '', 'note': '', 'label': f'{name} | {rate} kHz | HiFi-GAN',
    }


CATALOG = tuple(sorted(
    (*CATALOG, *(_politrees_entry(*entry) for entry in POLITREES_PRETRAINS)),
    key=lambda entry: entry['label'].casefold(),
))


def choices():
    return [entry['label'] for entry in CATALOG]


def entry_for(label):
    return next((entry for entry in CATALOG if entry['label']==label), None)


def urls(entry):
    prefix='https://huggingface.co/'+entry['repo']+'/resolve/main/'
    return tuple(prefix+quote(entry[key],safe='/')+'?download=true' for key in ('g','d'))


def describe(label):
    entry=entry_for(label)
    if entry is None:return 'Choose a recommended pretrained to see its sample rate and download files.'
    g,d=urls(entry)
    sample_rate = '%g kHz' % (entry['sample_rate'] / 1000) if entry['sample_rate'] else 'Not specified'
    rows=['| Setting | Value |','| --- | --- |','| Pretrained | '+entry['name']+' |',
          '| Sample rate | '+sample_rate+' |',
          '| Vocoder | '+entry['vocoder']+' |','| Embedder | '+entry['embedder']+' |']
    if entry['variant']:rows.append('| Variant | '+entry['variant']+' |')
    rows+=['','[Generator G]('+g+') · [Discriminator D]('+d+') · [Publisher page](https://huggingface.co/'+entry['repo']+')']
    if entry['vocoder']=='RefineGAN' and entry['sample_rate']!=32000:
        rows+=['','**Download available; training unsupported here:** this RefineGAN variant is %g kHz. This build trains RefineGAN only at 32 kHz. Do not select 32 kHz to disguise a different-rate pretrained.'%(entry['sample_rate']/1000)]
    if entry['note']:rows+=['',entry['note']]
    return '\n'.join(rows)


def download(label,root):
    from ilaria_pretrain_download import download_pretrained
    entry=entry_for(label)
    if entry is None:
        yield '0.0% | Select a recommended pretrained first.'
        return
    yield f"0.0% | Selected {entry['label']}. Downloading both G and D..."
    g,d=urls(entry)
    yield from download_pretrained(g,root,d)
