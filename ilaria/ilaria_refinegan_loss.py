"""RefineGAN training components from IAHispano/Applio, adapted for PyTorch 2.0.
https://github.com/IAHispano/Applio

MIT License

Copyright (c) 2026 AI Hispano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import torch
from librosa.filters import mel as librosa_mel_fn

class MultiScaleMelSpectrogramLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: list[int] = [5, 10, 20, 40, 80, 160, 320],  # , 480],
        window_lengths: list[int] = [32, 64, 128, 256, 512, 1024, 2048],  # , 4096],
        loss_fn=torch.nn.L1Loss(),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params: list[tuple] = []
        self.hann_window: dict[int, torch.Tensor] = {}
        self.mel_banks: dict[int, torch.Tensor] = {}

        self.stft_params = [(mel, win) for mel, win in zip(n_mels, window_lengths)]

    def mel_spectrogram(
        self,
        wav: torch.Tensor,
        n_mels: int,
        window_length: int,
    ):
        if wav.device.type == "mps":
            return self.mel_spectrogram(wav.cpu(), n_mels, window_length).to(wav.device)
        # IDs for caching
        dtype_device = str(wav.dtype) + "_" + str(wav.device)
        win_dtype_device = str(window_length) + "_" + dtype_device
        mel_dtype_device = str(n_mels) + "_" + dtype_device
        # caching hann window
        if win_dtype_device not in self.hann_window:
            self.hann_window[win_dtype_device] = torch.hann_window(
                window_length, device=wav.device, dtype=torch.float32
            )

        wav = wav.squeeze(1)  # -> torch(B, T)

        stft = torch.stft(
            wav.float(),
            n_fft=window_length,
            hop_length=window_length // 4,
            window=self.hann_window[win_dtype_device],
            return_complex=True,
        )  # -> torch (B, window_length // 2 + 1, (T - window_length)/hop_length + 1)

        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

        # caching mel filter
        if mel_dtype_device not in self.mel_banks:
            self.mel_banks[mel_dtype_device] = torch.from_numpy(
                librosa_mel_fn(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None,
                )
            ).to(device=wav.device, dtype=torch.float32)

        mel_spectrogram = torch.matmul(
            self.mel_banks[mel_dtype_device], magnitude
        )  # torch(B, n_mels, stft.frames)
        return mel_spectrogram

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ):  # real: torch(B, 1, T) , fake: torch(B, 1, T)
        loss = 0.0
        for p in self.stft_params:
            real_mels = self.mel_spectrogram(real, *p)
            fake_mels = self.mel_spectrogram(fake, *p)
            real_logmels = torch.log(real_mels.clamp(min=1e-5)) / self.log_base
            fake_logmels = torch.log(fake_mels.clamp(min=1e-5)) / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)
        return loss
