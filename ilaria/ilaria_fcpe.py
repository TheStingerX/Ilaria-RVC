"""Bridge the official TorchFCPE model to RVC's 16 kHz, 10 ms pitch grid."""

from pathlib import Path
import hashlib
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "assets" / "pitch" / "fcpe_runtime"
MODEL_SHA256 = "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015"
SAMPLE_RATE = 16000
HOP = 160


def fcpe_device(device):
    # FCPE's STFT path uses complex tensors. Use CPU on other backends.
    return str(device) if str(device).startswith("cuda") else "cpu"


class FCPEPitch:
    def __init__(self, device):
        self.device = torch.device(fcpe_device(device))
        model_file = VENDOR / "torchfcpe" / "assets" / "fcpe_c_v001.pt"
        if not model_file.is_file():
            raise RuntimeError("The FCPE model is missing. Restore assets/pitch/fcpe_runtime.")
        digest = hashlib.sha256()
        with model_file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError("The FCPE model failed verification. Restore assets/pitch/fcpe_runtime.")
        vendor = str(VENDOR)
        if vendor not in sys.path:
            sys.path.append(vendor)
        from torchfcpe.models_infer import InferCFNaiveMelPE
        from torchfcpe.tools import DotDict

        checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=True)
        options = checkpoint["config_dict"]
        options["model"].update(conv_dropout=0.0, atten_dropout=0.0)
        options["model"].setdefault("use_harmonic_emb", False)
        self.model = InferCFNaiveMelPE(DotDict(options), checkpoint["model"]).to(self.device).eval()

    @torch.inference_mode()
    def get_f0(self, audio, frames):
        if frames <= 0:
            return np.zeros(0, dtype=np.float64)
        # Long tracks are decoded in chunks with surrounding context.
        chunk_frames = 3000
        context_frames = 25
        result = np.zeros(frames, dtype=np.float64)
        for first in range(0, frames, chunk_frames):
            last = min(frames, first + chunk_frames)
            start = max(0, first - context_frames)
            stop = min(frames, last + context_frames)
            segment = audio[start * HOP:stop * HOP]
            required = max(1024, (stop - start) * HOP)
            if len(segment) < required:
                segment = np.pad(segment, (0, required - len(segment)))
            waveform = torch.as_tensor(segment, device=self.device, dtype=torch.float32)[None, :, None]
            pitch = self.model.infer(
                waveform, sr=SAMPLE_RATE, decoder_mode="local_argmax",
                threshold=0.006, f0_min=50, f0_max=1100, interp_uv=False,
            ).reshape(-1).cpu().numpy()
            result[first:last] = pitch[first - start:last - start]
        result[(result < 50) | ~np.isfinite(result)] = 0
        return result
