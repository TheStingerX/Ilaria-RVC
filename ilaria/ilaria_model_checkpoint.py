"""Export an inference voice model, preserving vocoder/embedder metadata."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from collections import OrderedDict
import os
import tempfile
import traceback
import torch

def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        opt = OrderedDict()
        opt['weight'] = {}
        for key in ckpt.keys():
            if 'enc_q' in key:
                continue
            opt['weight'][key] = ckpt[key].half()
        opt['config'] = [hps.data.filter_length // 2 + 1, 32, hps.model.inter_channels, hps.model.hidden_channels, hps.model.filter_channels, hps.model.n_heads, hps.model.n_layers, hps.model.kernel_size, hps.model.p_dropout, hps.model.resblock, hps.model.resblock_kernel_sizes, hps.model.resblock_dilation_sizes, hps.model.upsample_rates, hps.model.upsample_initial_channel, hps.model.upsample_kernel_sizes, hps.model.spk_embed_dim, hps.model.gin_channels, hps.data.sampling_rate]
        opt['info'] = '%sepoch' % epoch
        opt['sr'] = sr
        opt['f0'] = if_f0
        opt['version'] = version
        from ilaria_vocoder import architecture_metadata
        opt.update(architecture_metadata(hps))
        folder = str(getattr(hps, "name", None) or name)
        if any(not value or value in (".", "..") or _IlariaBootstrapPath(value).name != value
               for value in (folder, name)):
            raise ValueError("Use a model name without folder paths.")
        directory = _ilaria_install_root / "models" / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".pth")
        fd, temporary = tempfile.mkstemp(prefix=".model-", suffix=".partial", dir=directory)
        os.close(fd)
        try:
            with open(temporary, "wb") as stream:
                torch.save(opt, stream)
            os.replace(temporary, path)
        finally:
            _IlariaBootstrapPath(temporary).unlink(missing_ok=True)
        return 'Success.'
    except:
        return traceback.format_exc()
