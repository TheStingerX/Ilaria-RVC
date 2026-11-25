import re
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


KLM_PRETRAIN_STEMS = {
    'KLM4': (
        'KLM40_32k.pth',
        'KLM40_40k.pth',
        'KLM40_48k.pth',
        'KLM41_T2_32k.pth',
    ),
    'KLM4.1': (
        'KLM41_32k.pth',
        'KLM41_48k.pth',
    ),
    'KLM4.2': (
        'KLM42_32k_x10.pth',
    ),
    'KLM43': (
        'KLM43_x1_32k.pth',
        'KLM43_x2_32k.pth',
        'KLM43_X3_32k.pth',
        'KLM43_X3_40k.pth',
        'KLM43_X3_48k.pth',
        'KLM43_x4_32k.pth',
        'KLM43_x4_40k.pth',
        'KLM43_x4_48k.pth',
    ),
    'KLM4.4': (
        'KLM44_X1_32Khz.pth',
        'KLM44_X2_32k.pth',
        'KLM44_X2B_32k.pth',
    ),
    'KLM6_Experimental': (
        'KLM6exp_32khz.pth',
        'KLM6exp_48khz.pth',
        'KLM6_Exp2_32khz.pth',
        'KLM6_Exp2_40khz.pth',
        'Dev_KLM61Exp_48khz.pth',
        'Dev_KLM61Exp_Ranger25_48khz.pth',
    ),
    'RVC_KLM9_HFG_PretrainModel': (
        'KLM9T_HFG_32k.pth',
    ),
    'RVC_KLM-X-10_Pretrained-Model': (
        'KLM-X_Base_32k.pth',
    ),
    'KLM_RVC_KLM-HF_Trainer': (
        'KLM-X1_32khz.pth',
        'KLM_HF_Mini_T_32k.pth',
        'KLM_RVC_PT_32k.pth',
        'KLM_RVC_PT_40k.pth',
        'KLM_RVC_PT_48k.pth',
        'RVC_OG_32k.pth',
        'RVC_OG_40k.pth',
        'RVC_OG_48k.pth',
    ),
    'KLM_KpopUniverse': (
        'KLM_KpopUniverse_32k.pth',
        'KLM_KpopUniverse_40k.pth',
        'KLM_KpopUniverse_48k.pth',
        'KLM_KPU_II_32k.pth',
        'KLM_KPU_II_40k.pth',
    ),
    'BlackTone_RVC_Pretrained': (
        'BlackTone_32k.pth',
        'Large_BlackTone_32k.pth',
    ),
    'KLM_BEATMASTER': (
        'KLM_BeatMasterV1_48k.pth',
        'KLM_BeatzForge_48k.pth',
    ),
}


def _klm_entry(repository, stem):
    rate_match = re.search(r'_(32|40|48)k(?:hz)?(?=_|\.pth$)', stem, re.IGNORECASE)
    rate = int(rate_match[1])
    name = (stem[:rate_match.start()] + stem[rate_match.end():-4]).replace('_', ' ')
    vocoder = 'HiFi-GAN' if '_HFG_' in stem else 'Not specified'
    return {
        'name': name, 'repo': 'SeoulStreamingStation/' + repository,
        'g': 'G_' + stem, 'd': 'D_' + stem,
        'sample_rate': rate * 1000, 'vocoder': vocoder, 'embedder': 'Not specified',
        'variant': '', 'label': f'{name} | {rate} kHz',
        'note': 'Publisher G/D files verified. Training compatibility has not been tested in Ilaria RVC.',
    }


CATALOG = tuple(sorted(
    (*CATALOG, *(_klm_entry(repo, stem) for repo, stems in KLM_PRETRAIN_STEMS.items() for stem in stems)),
    key=lambda entry: entry['label'].casefold(),
))


LEGACY_CORE_PRETRAINS = (
    ('Legacy Core 1.5', 'sxndypz/legacy-core-1.5', '15', 32, 'HiFi-GAN', 'contentvec'),
    ('Legacy Core 1.5', 'sxndypz/legacy-core-1.5', '40k', 40, 'HiFi-GAN', 'contentvec'),
    ('Legacy Core 1.5 V2', 'notiscBalbalEko/PretrainReupload', 'LegacyCore1.5v2', 32, 'HiFi-GAN', 'contentvec'),
    ('Legacy Core 1.6', 'notiscBalbalEko/PretrainReupload', 'LegacyCoreOld', 32, 'HiFi-GAN', 'contentvec'),
)

CATALOG = tuple(sorted(
    (*CATALOG, *(
        {
            'name': name, 'repo': repo,
            'g': f'G_{stem}.pth', 'd': f'D_{stem}.pth',
            'sample_rate': rate * 1000, 'vocoder': vocoder, 'embedder': embedder,
            'variant': '', 'note': "Reupload of Lyery's pretrained weights.",
            'label': f'{name} | {rate} kHz',
        }
        for name, repo, stem, rate, vocoder, embedder in LEGACY_CORE_PRETRAINS
    )),
    key=lambda entry: entry['label'].casefold(),
))


DOCS_PRETRAIN_FILES = (
    ('Stock RVC v1 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/f0G32k.pth', 'pretrained/f0D32k.pth', 32, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v1 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/G32k.pth', 'pretrained/D32k.pth', 32, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v1 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/f0G40k.pth', 'pretrained/f0D40k.pth', 40, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v1 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/G40k.pth', 'pretrained/D40k.pth', 40, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v1 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/f0G48k.pth', 'pretrained/f0D48k.pth', 48, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v1 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained/G48k.pth', 'pretrained/D48k.pth', 48, 'HiFi-GAN', 'HuBERT Base (256)', 'Requires RVC v1 training support.'),
    ('Stock RVC v2 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/f0G32k.pth', 'pretrained_v2/f0D32k.pth', 32, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Stock RVC v2 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/G32k.pth', 'pretrained_v2/D32k.pth', 32, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Stock RVC v2 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/f0G40k.pth', 'pretrained_v2/f0D40k.pth', 40, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Stock RVC v2 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/G40k.pth', 'pretrained_v2/D40k.pth', 40, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Stock RVC v2 (F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/f0G48k.pth', 'pretrained_v2/f0D48k.pth', 48, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Stock RVC v2 (No F0)', 'lj1995/VoiceConversionWebUI', 'pretrained_v2/G48k.pth', 'pretrained_v2/D48k.pth', 48, 'HiFi-GAN', 'HuBERT Base (768)', ''),
    ('Aurora v2', 'Politrees/RVC_resources', 'pretrained/v2/48k/Aurora/G_Aurora_v2.pth', 'pretrained/v2/48k/Aurora/D_Aurora_v2.pth', 48, 'Not specified', 'Not specified', ''),
    ('DMR v0.5', 'Politrees/RVC_resources', 'pretrained/v2/32k/DMR/G_dmrV0-5.pth', 'pretrained/v2/32k/DMR/D_dmrV0-5.pth', 32, 'Not specified', 'Not specified', ''),
    ('DMR v0.8', 'Politrees/RVC_resources', 'pretrained/v2/32k/DMR/G_DMR_V0-8.pth', 'pretrained/v2/32k/DMR/D_DMR_V0-8.pth', 32, 'Not specified', 'Not specified', ''),
    ('DMR v1', 'Politrees/RVC_resources', 'pretrained/v2/32k/DMR/G_DMR-V1.pth', 'pretrained/v2/32k/DMR/D_DMR-V1.pth', 32, 'Not specified', 'Not specified', ''),
    ('SingerPreTrain v1', 'Politrees/RVC_resources', 'pretrained/v2/32k/Singer/f0G_SingerPreTrain.pth', 'pretrained/v2/32k/Singer/f0D_SingerPreTrain.pth', 32, 'Not specified', 'Not specified', ''),
    ('KLM 4.2 T1', 'Politrees/RVC_resources', 'pretrained/v2/32k/KLM/G_KLM42_T1_32k.pth', 'pretrained/v2/32k/KLM/D_KLM42_T1_32k.pth', 32, 'Not specified', 'Not specified', ''),
    ('KLM 4.2 T1', 'Politrees/RVC_resources', 'pretrained/v2/40k/KLM/G_KLM42_fp32_T1_40k.pth', 'pretrained/v2/40k/KLM/D_KLM42_fp32_T1_40k.pth', 40, 'Not specified', 'Not specified', ''),
    ('KLM 4.2 T2', 'Politrees/RVC_resources', 'pretrained/v2/40k/KLM/G_KLM42_fp32_T2_40k.pth', 'pretrained/v2/40k/KLM/D_KLM42_fp32_T2_40k.pth', 40, 'Not specified', 'Not specified', ''),
    ('KLM 4.2 T3', 'Politrees/RVC_resources', 'pretrained/v2/40k/KLM/G_KLM42_fp32_T3_40k.pth', 'pretrained/v2/40k/KLM/D_KLM42_fp32_T3_40k.pth', 40, 'Not specified', 'Not specified', ''),
    ('KLM 4.2 T4', 'Politrees/RVC_resources', 'pretrained/v2/40k/KLM/G_KLM42_T4_40k.pth', 'pretrained/v2/40k/KLM/D_KLM42_T4_40k.pth', 40, 'Not specified', 'Not specified', ''),
    ('Ov2Super F0', 'Politrees/RVC_resources', 'pretrained/v2/40k/Ov2/f0Ov2Super40kG.pth', 'pretrained/v2/40k/Ov2/f0Ov2Super40kD.pth', 40, 'HiFi-GAN', 'Not specified', ''),
    ('Nanashi v1 original', 'shiromiya/nanashi-pretrain', 'v1/G_nanashi_v1.pth', 'v1/D_nanashi_v1.pth', 32, 'Not specified', 'Not specified', ''),
    ('Nanashi v1.5', 'shiromiya/nanashi-pretrain', 'v1.5/G_nanashi_v1_5.pth', 'v1.5/D_nanashi_v1_5.pth', 32, 'Not specified', 'Not specified', ''),
    ('Nanashi v1.5 Rigel', 'shiromiya/nanashi-pretrain', 'v1.5_Rigel_Version/G_Rigel_Portuguese_50e.pth', 'v1.5_Rigel_Version/D_Rigel_Portuguese_50e.pth', 32, 'Not specified', 'Not specified', ''),
    ('Nanashi Anime Resized', 'shiromiya/nanashi-pretrain', 'v1_anime/resized/G_nanashi_anime_resized.pth', 'v1_anime/resized/D_nanashi_anime_resized.pth', 32, 'Not specified', 'Not specified', ''),
    ('Rigel Base', 'MUSTAR/Rigel-rvc-base-pretrained-model', 'Rigel_40k_Base_and_FineTuned/Base-model_40k/Alpha/G_RIGEL_base_40k_371406.pth', 'Rigel_40k_Base_and_FineTuned/Base-model_40k/Alpha/D_RIGEL_base_40k_371406.pth', 40, 'HiFi-GAN', 'Not specified', ''),
    ('Rigel Fine-tuned', 'MUSTAR/Rigel-rvc-base-pretrained-model', 'Rigel_40k_Base_and_FineTuned/FineTuned-model_40k/Alpha/G_RigelFT_40k_9801520.pth', 'Rigel_40k_Base_and_FineTuned/FineTuned-model_40k/Alpha/D_RigelFT_40k_9801520.pth', 40, 'HiFi-GAN', 'Not specified', ''),
    ('Sench Pretrain', 'Shashashasha/Sench_Pretrain_48k_RVC', 'G_2333333.pth', 'D_2333333.pth', 48, 'Not specified', 'Not specified', ''),
    ('Snowie v3.1 Japanese HuBERT', 'MUSTAR/Snowie_V3.1_pretrain_on_JPhubert', 'G_SnowieV3.1_JP_hubert.pth', 'D_SnowieV3.1_JP_hubert.pth', None, 'Not specified', 'Japanese HuBERT', ''),
    ('Snowie RuPretrain (No F0)', 'MUSTAR/Snowie_RuPretrain_noP_40k', 'G_Snow_RuPretrain_noP.pth', 'D_Snow_RuPretrain_noP.pth', 40, 'Not specified', 'Not specified', 'Requires Pitch Guidance off.'),
    ('DeepFake Audio Rangers', 'DeepFake-Audio-Rangers/rvc_pretrain', 'G_45780.pth', 'D_45780.pth', None, 'Not specified', 'Not specified', ''),
    ('Applio RefineGAN (DMS)', 'DanielMarius/dms-applio-refinegan-pretrained', 'f0G24k.pth', 'f0D24k.pth', 24, 'RefineGAN', 'Not specified', 'Download only: 24 kHz training is unavailable.'),
    ('Applio RefineGAN (DMS)', 'DanielMarius/dms-applio-refinegan-pretrained', 'f0G32k.pth', 'f0D32k.pth', 32, 'RefineGAN', 'Not specified', 'Training compatibility has not been verified.'),
    ('RingFormer v1', 'its5Q/rvc-ringformer-pretrain', 'final/G_its5Q.pth', 'final/D_its5Q.pth', 40, 'RingFormer', 'SPIN v2', 'Archive reference. RingFormer training is unavailable in this build.'),
    ('RingFormer v2 (failed)', 'its5Q/rvc-ringformer-v2-pretrain', 'failed/G_481376.pth', 'failed/D_481376.pth', 40, 'RingFormer v2', 'SPIN v2', 'Archive reference. RingFormer training is unavailable in this build.'),
    ('Convbased HiFi-GAN (contentvec)', 'PluginsKers/Convbased-Studio', 'contentvec/HiFiGAN/476sid/f0G48k.pth', 'contentvec/HiFiGAN/476sid/f0D48k.pth', 48, 'HiFi-GAN', 'contentvec', 'Training compatibility has not been verified.'),
    ('Convbased SiFi-GAN (contentvec)', 'PluginsKers/Convbased-Studio', 'contentvec/SiFiGAN/f0G48k.pth', 'contentvec/SiFiGAN/f0D48k.pth', 48, 'SiFi-GAN', 'contentvec', 'Download reference. This vocoder is unavailable in Ilaria RVC.'),
    ('Convbased BigVGAN (contentvec)', 'PluginsKers/Convbased-Studio', 'contentvec/BigVGAN/f0G40k.pth', 'contentvec/BigVGAN/f0D40k.pth', 40, 'BigVGAN', 'contentvec', 'Download reference. This vocoder is unavailable in Ilaria RVC.'),
    ('Convbased HiFi-GAN (spin)', 'PluginsKers/Convbased-Studio', 'spin/HiFiGAN/f0G48k.pth', 'spin/HiFiGAN/f0D48k.pth', 48, 'HiFi-GAN', 'spin', 'Training compatibility has not been verified.'),
    ('Convbased SiFi-GAN (spin)', 'PluginsKers/Convbased-Studio', 'spin/SiFiGAN/f0G48k.pth', 'spin/SiFiGAN/f0D48k.pth', 48, 'SiFi-GAN', 'spin', 'Download reference. This vocoder is unavailable in Ilaria RVC.'),
    ('Convbased BigVGAN (spin-v2)', 'PluginsKers/Convbased-Studio', 'spin-v2/BigVGAN/f0G40k.pth', 'spin-v2/BigVGAN/f0D40k.pth', 40, 'BigVGAN', 'spin-v2', 'Download reference. This vocoder is unavailable in Ilaria RVC.'),
    ('Convbased HiFi-GAN (chinese-hubert-base)', 'PluginsKers/Convbased-Studio', 'chinese-hubert-base/HiFiGAN/f0G48k.pth', 'chinese-hubert-base/HiFiGAN/f0D48k.pth', 48, 'HiFi-GAN', 'chinese-hubert-base', 'Training compatibility has not been verified.'),
    ('7subpretrain', 'makiligon/rvc-sub-pretrain-moment', '7subpretrain40k.zip', '', 40, 'Not specified', 'Not specified', 'Experimental archive. The installer checks for a complete G/D pair before installing.'),
    ('Genshin EN Pretrain', 'makiligon/rvc-sub-pretrain-moment', 'GenshinENPretrain32K.zip', '', 32, 'Not specified', 'Not specified', 'Experimental archive. The installer checks for a complete G/D pair before installing.'),
    ('Genshin EN Pretrain', 'makiligon/rvc-sub-pretrain-moment', 'GenshinENPretrain40k.zip', '', 40, 'Not specified', 'Not specified', 'Experimental archive. The installer checks for a complete G/D pair before installing.'),
    ('SingingPretrain', 'makiligon/rvc-sub-pretrain-moment', 'SingingPretrain.zip', '', None, 'Not specified', 'Not specified', 'Experimental archive. The installer checks for a complete G/D pair before installing.'),
    ('RVC v2 Pretrain Voice', 'Coolwowsocoolwow/RVC_V2_Pretrain_Voice', 'RVC_V2_Pretrain_Voice.zip', '', None, 'Not specified', 'Not specified', 'Experimental archive. The installer checks for a complete G/D pair before installing.'),
)


def _docs_pretrain_entries():
    for name, repo, generator, discriminator, rate, vocoder, embedder, note in DOCS_PRETRAIN_FILES:
        rate_label = f'{rate} kHz' if rate is not None else 'Sample rate unconfirmed'
        yield {
            'name': name, 'repo': repo, 'g': generator, 'd': discriminator,
            'sample_rate': rate * 1000 if rate is not None else None,
            'vocoder': vocoder, 'embedder': embedder, 'variant': '', 'note': note,
            'label': f'{name} | {rate_label}',
        }


CATALOG = tuple(sorted((*CATALOG, *_docs_pretrain_entries()), key=lambda entry: entry['label'].casefold()))


def choices():
    return [entry['label'] for entry in CATALOG]


def entry_for(label):
    return next((entry for entry in CATALOG if entry['label']==label), None)


def urls(entry):
    prefix='https://huggingface.co/'+entry['repo']+'/resolve/main/'
    return tuple(prefix+quote(entry[key],safe='/')+'?download=true' for key in ('g','d') if entry[key])


def describe(label):
    entry=entry_for(label)
    if entry is None:return 'Choose a recommended pretrained to see its sample rate and download files.'
    links = urls(entry)
    sample_rate = '%g kHz' % (entry['sample_rate'] / 1000) if entry['sample_rate'] else 'Not specified'
    rows=['| Setting | Value |','| --- | --- |','| Pretrained | '+entry['name']+' |',
          '| Sample rate | '+sample_rate+' |',
          '| Vocoder | '+entry['vocoder']+' |','| Embedder | '+entry['embedder']+' |']
    if entry['variant']:rows.append('| Variant | '+entry['variant']+' |')
    downloads = (
        f'[Generator G]({links[0]}) · [Discriminator D]({links[1]})'
        if entry['d'] else f'[Pretrained ZIP]({links[0]})'
    )
    rows += ['', downloads + ' · [Publisher page](https://huggingface.co/' + entry['repo'] + ')']
    if entry['vocoder']=='RefineGAN' and entry['sample_rate'] in (40000, 44100):
        rows+=['','**Architecture differs:** this %g kHz pretrained pair can be downloaded, but does not match the RefineGAN decoder in Ilaria RVC. Choose None to train from scratch.'%(entry['sample_rate']/1000)]
    if entry['note']:rows+=['',entry['note']]
    return '\n'.join(rows)


def download(label,root):
    from ilaria_pretrain_download import download_pretrained
    entry=entry_for(label)
    if entry is None:
        yield '0.0% | Select a recommended pretrained first.'
        return
    kind = 'both G and D' if entry['d'] else 'the pretrained ZIP'
    yield f"0.0% | Selected {entry['label']}. Downloading {kind}..."
    source, *discriminator = urls(entry)
    yield from download_pretrained(source, root, discriminator[0] if discriminator else '')
