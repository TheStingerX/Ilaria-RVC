
import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[3]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import argparse
import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def parse_extraction_args(argv):
    parser = argparse.ArgumentParser(description="Extract HuBERT training features.")
    parser.add_argument("device")
    parser.add_argument("n_part", type=int)
    parser.add_argument("i_part", type=int)
    parser.add_argument("paths", nargs="+", metavar="[GPU] EXP_DIR VERSION")
    parser.add_argument("--embedder", choices=("contentvec", "spin-v2"), default="contentvec")
    parser.add_argument("--precision", choices=("FP16", "FP32"), default="FP32")
    args = parser.parse_args(argv)
    if len(args.paths) == 2:
        args.i_gpu = None
        args.exp_dir, args.version = args.paths
    elif len(args.paths) == 3:
        args.i_gpu, args.exp_dir, args.version = args.paths
    else:
        parser.error("expected DEVICE N_PART I_PART [GPU] EXP_DIR VERSION")
    return args


args = parse_extraction_args(sys.argv[1:])
device = args.device
n_part = args.n_part
i_part = args.i_part
exp_dir = args.exp_dir
version = args.version
if args.i_gpu is not None:
    i_gpu = args.i_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(sys.argv)
requested_precision = args.precision
effective_precision = (
    "FP16"
    if requested_precision == "FP16"
    and str(device).split(":", 1)[0] == "cuda"
    and torch.cuda.is_available()
    else "FP32"
)
use_half = effective_precision == "FP16"
printt(
    "Feature extraction precision: requested=%s, effective=%s, device=%s"
    % (requested_precision, effective_precision, device)
)
if requested_precision != effective_precision:
    printt("FP16 feature extraction requires CUDA; using FP32 on this device.")
model_path = "assets/hubert/hubert_base.pt"

printt(exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
from ilaria_embedders import load_embedder
model, saved_cfg = load_embedder(os.getcwd(), args.embedder, device, use_half)
printt("move model to %s" % device)
if args.embedder == "spin-v2" and i_part == 0:
    # Silence must use the same representation as the voice recordings.
    silent = readwave("logs/mute/1_16k_wavs/mute.wav", normalize=saved_cfg.task.normalize)
    with torch.no_grad():
        value = model.extract_features(source=(silent.half() if use_half else silent.float()).to(device),
            padding_mask=torch.zeros_like(silent, dtype=torch.bool).to(device), output_layer=9 if version == "v1" else 12)[0]
        value = model.final_proj(value) if version == "v1" else value
    silent_folder = "logs/mute_spin-v2/3_feature%s" % (256 if version == "v1" else 768)
    os.makedirs(silent_folder, exist_ok=True)
    np.save(silent_folder + "/mute.npy", value.squeeze(0).float().cpu().numpy(), allow_pickle=False)

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (feats.half() if use_half else feats.float()).to(device),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
