"""Adapted from redpanda343/Redpanda-vc, commit55a3a806.
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
from __future__ import annotations
import functools
import math
try:
    import tomllib
except ImportError:
    tomllib = None
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VERSION = '2.0.0-rc.0'
IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 24000
IN_HOP = 160
OUT_HOP = 240
PITCH_BINS = 448
BINS_PER_OCTAVE = 96
PITCH_BAND = 4
PITCH_BASE_HZ = 55.0
PHONE_CHANNELS = 128
HIDDEN_CHANNELS = 256
CODEBOOK_SIZE = 512
KV_LENGTH = 384
KV_CHANNELS = 128
HEADS = 4
FORMANT_STEPS = 9
ATTENTION_FRAMES = 400
PITCH_WINDOW = 560
CORRELATION_WINDOW = 304
INSTFREQ_BINS = 64
IR_LENGTH = 512
FILTER_FFT = 768
PHONE_LOOKAHEAD = 40
HISTORY = 2 * IN_HOP + (PITCH_WINDOW - IN_HOP) // 2
LATENCY_SECONDS = 0.0375


def is_beatrice_checkpoint(path):
    path = Path(path)
    return path.name.startswith('checkpoint_') and path.name.endswith(('.pt.gz', '.pt'))


def find_paraphernalia(path):
    path = Path(path)
    if path.is_file() and path.suffix == '.toml':
        return path
    if is_beatrice_checkpoint(path):
        path = path.parent
    if path.is_dir():
        found = sorted(path.glob('beatrice_paraphernalia_*.toml'))
        if len(found) == 1:
            return found[0]
        folders = sorted((folder for folder in path.glob('paraphernalia_*') if folder.is_dir()),
                         key=lambda folder: folder.stat().st_mtime, reverse=True)
        for folder in folders:
            found = sorted(folder.glob('beatrice_paraphernalia_*.toml'))
            if len(found) == 1:
                return found[0]
    return None


def read_paraphernalia(path):
    toml = find_paraphernalia(path)
    if toml is None:
        raise FileNotFoundError(f'No beatrice_paraphernalia_*.toml model found at {path}.')
    with toml.open('rb') as handle:
        config = tomllib.load(handle)
    version = config.get('model', {}).get('version')
    if version != VERSION:
        raise ValueError(f'This Beatrice model has format {version}; only {VERSION} models are supported.')
    voices = config.get('voice', {})
    voices = [voices[key] for key in sorted(voices, key=int)]
    return toml.parent, config['model'].get('name', toml.parent.name), voices


class Weights:
    def __init__(self, path, device, values=None):
        self.path = Path(path)
        if values is None:
            if not self.path.is_file():
                raise FileNotFoundError(f'Beatrice model file not found: {self.path}')
            values = np.fromfile(self.path, dtype=np.float16)
        self.values = torch.from_numpy(values.astype(np.float32)).to(device)
        self.offset = 0

    def take(self, *shape):
        count = math.prod(shape)
        if self.offset + count > self.values.numel():
            raise ValueError(f'{self.path.name} is too small for a Beatrice {VERSION} model.')
        tensor = self.values[self.offset:self.offset + count].reshape(shape)
        self.offset += count
        return tensor

    def finish(self):
        if self.offset != self.values.numel():
            raise ValueError(f'{self.path.name} does not match the Beatrice {VERSION} layout.')


def checkpoint_pitch_values(path):
    state = torch.load(path, map_location='cpu', weights_only=True)['pitch_estimator']
    tensors = []
    for name in ('instfreq_embed_0', 'instfreq_embed_1', 'corr_embed_0', 'corr_embed_1', 'backbone.embed',
                 'backbone.norm'):
        tensors += [state[f'{name}.weight'], state[f'{name}.bias']]
    block = 0
    while f'backbone.convnext.{block}.gamma' in state:
        prefix = f'backbone.convnext.{block}.'
        scale = state[prefix + 'post_scale'] * state[prefix + 'post_scale_weight']
        tensors += [
            state[prefix + 'dwconv.weight'] * state[prefix + 'pre_scale'],
            state[prefix + 'dwconv.bias'],
            state[prefix + 'pwconv1.weight'] * state[prefix + 'norm.weight'][None],
            state[prefix + 'pwconv1.bias'] + torch.mv(state[prefix + 'pwconv1.weight'], state[prefix + 'norm.bias']),
            state[prefix + 'pwconv2.weight'] * state[prefix + 'gamma'][:, None] * scale,
            state[prefix + 'pwconv2.bias'] * state[prefix + 'gamma'] * scale,
        ]
        block += 1
    for name in ('backbone.final_layer_norm', 'head'):
        tensors += [state[f'{name}.weight'], state[f'{name}.bias']]
    return torch.cat([tensor.reshape(-1) for tensor in tensors]).half().numpy()


@functools.lru_cache(maxsize=64)
def phase_mask(frames, keys, device):
    distance = (torch.arange(frames, device=device)[:, None] - torch.arange(keys, device=device)[None]
                + keys - frames)
    return (distance >= 0) & (distance % 4 == 0) & (distance < ATTENTION_FRAMES)


class Conv:
    def __init__(self, weights, in_channels, out_channels, kernel_size, stride=1, depthwise=False, bias=True):
        weight = weights.take(out_channels, 1 if depthwise else in_channels, kernel_size)
        self.weight = weight[:, 0] if depthwise else weight.reshape(out_channels, in_channels * kernel_size)
        self.bias = weights.take(out_channels) if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.depthwise = depthwise
        self.history = None

    def reset(self):
        self.history = None

    def __call__(self, x):
        if self.kernel_size > 1:
            if self.stride == 1:
                if self.history is None:
                    self.history = x.new_zeros(self.kernel_size - 1, x.shape[1])
                x = torch.cat([self.history, x])
                self.history = x[-(self.kernel_size - 1):]
            windows = x.unfold(0, self.kernel_size, self.stride)
            if self.depthwise:
                output = (windows * self.weight).sum(-1)
                return output if self.bias is None else output + self.bias
            x = windows.reshape(windows.shape[0], -1)
        return F.linear(x, self.weight, self.bias)


class PhaseAttention:
    def __init__(self, weights, channels):
        head = channels // HEADS
        self.qkv_weight = weights.take(HEADS, 3, head, channels).transpose(0, 1).reshape(3 * channels, channels)
        self.qkv_bias = weights.take(HEADS, 3, head).transpose(0, 1).reshape(3 * channels)
        self.out_weight = weights.take(channels, channels)
        self.out_bias = weights.take(channels)
        self.reset()

    def reset(self):
        self.keys = None
        self.values = None

    def __call__(self, x):
        frames, channels = x.shape
        query, key, value = F.linear(x, self.qkv_weight, self.qkv_bias).view(frames, 3, HEADS, -1).unbind(1)
        if self.keys is not None:
            key = torch.cat([self.keys, key])
            value = torch.cat([self.values, value])
        self.keys = key[-ATTENTION_FRAMES:]
        self.values = value[-ATTENTION_FRAMES:]
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
            attn_mask=phase_mask(frames, key.shape[0], x.device), scale=1.0,
        )
        return F.linear(output.transpose(0, 1).reshape(frames, channels), self.out_weight, self.out_bias)


class SpeakerAttention:
    def __init__(self, weights, channels):
        self.query_weight = weights.take(KV_CHANNELS, channels)
        self.query_bias = weights.take(KV_CHANNELS)
        self.out_weight = weights.take(channels, KV_CHANNELS)
        self.out_bias = weights.take(channels)
        self.keys = None
        self.values = None

    def reset(self):
        pass

    def __call__(self, x):
        query = F.linear(x, self.query_weight, self.query_bias).view(x.shape[0], HEADS, -1).transpose(0, 1)
        output = F.scaled_dot_product_attention(query, self.keys, self.values, scale=1.0)
        return F.linear(output.transpose(0, 1).reshape(x.shape[0], KV_CHANNELS), self.out_weight, self.out_bias)


class Block:
    def __init__(self, weights, channels, intermediate, kernel_size, standardized, attention):
        self.attention = attention(weights, channels) if attention else None
        self.dwconv = Conv(weights, channels, channels, kernel_size, depthwise=True)
        self.pointwise1 = (weights.take(intermediate, channels), weights.take(intermediate))
        self.pointwise2 = (weights.take(channels, intermediate), weights.take(channels))
        self.standardized = standardized

    def reset(self):
        self.dwconv.reset()
        if self.attention is not None:
            self.attention.reset()

    def __call__(self, x):
        if self.attention is not None:
            x = x + self.attention(F.layer_norm(x, (x.shape[1],)))
        y = self.dwconv(x)
        if not self.standardized:
            y = F.layer_norm(y, (y.shape[1],))
        return x + F.linear(F.gelu(F.linear(y, *self.pointwise1), approximate='tanh'), *self.pointwise2)


class Stack:
    def __init__(self, weights, in_channels, channels, intermediate, blocks, embed_kernel, kernel_size,
                 standardized=False, attention=None):
        self.embed = Conv(weights, in_channels, channels, embed_kernel)
        self.norm = None if standardized else (weights.take(channels), weights.take(channels))
        self.blocks = [Block(weights, channels, intermediate, kernel_size, standardized, attention)
                       for _ in range(blocks)]
        self.final = None if standardized else (weights.take(channels), weights.take(channels))

    def reset(self):
        self.embed.reset()
        for block in self.blocks:
            block.reset()

    def __call__(self, x):
        x = self.embed(x)
        if self.norm is not None:
            x = F.layer_norm(x, (x.shape[1],), *self.norm)
        for block in self.blocks:
            x = block(x)
        if self.final is not None:
            x = F.layer_norm(x, (x.shape[1],), *self.final)
        return x


class PitchEstimator:
    def __init__(self, weights, device):
        self.instfreq_embed = (Conv(weights, 192, 192, 1), Conv(weights, 192, 192, 1))
        self.correlation_embed = (Conv(weights, 256, 192, 1), Conv(weights, 192, 192, 1))
        self.backbone = Stack(weights, 192, 192, 384, 9, 3, 33)
        self.head = Conv(weights, 192, PITCH_BINS, 1)
        weights.finish()
        self.cosine = torch.signal.windows.cosine(PITCH_WINDOW, device=device)
        self.bins = torch.arange(PITCH_BINS, device=device)

    def reset(self):
        for module in (*self.instfreq_embed, *self.correlation_embed, self.backbone, self.head):
            module.reset()

    def inputs(self, audio, frames):
        windows = audio.unfold(0, PITCH_WINDOW, IN_HOP)[:frames + 1]
        spectrum = torch.fft.rfft(windows, n=PITCH_WINDOW)[:, :INSTFREQ_BINS]
        delta = spectrum[1:] * spectrum[:-1].conj()
        delta = delta / delta.abs().add(1e-5)
        instfreq = torch.cat([spectrum[1:].abs().add(1e-5).log10(), delta.real, delta.imag], 1)
        windows = windows[1:]
        flipped = windows.flip(-1)
        correlation = torch.fft.irfft(
            torch.fft.rfft(flipped, n=PITCH_WINDOW) * torch.fft.rfft(windows[:, -CORRELATION_WINDOW:], n=PITCH_WINDOW),
            n=PITCH_WINDOW,
        )[:, CORRELATION_WINDOW:]
        energy = flipped.square().cumsum(-1)
        difference = (energy[:, CORRELATION_WINDOW - 1:CORRELATION_WINDOW] + energy[:, CORRELATION_WINDOW:]
                      - energy[:, :-CORRELATION_WINDOW] - 2.0 * correlation)
        difference = (difference.clamp(min=0.0) * (2.0 / CORRELATION_WINDOW)).sqrt()
        loudness = (windows * self.cosine).square().sum(-1).clamp(min=1e-3).log10() * 0.5
        return instfreq, difference, loudness

    def probabilities(self, instfreq, difference):
        instfreq = self.instfreq_embed[1](F.gelu(self.instfreq_embed[0](instfreq), approximate='tanh'))
        difference = self.correlation_embed[1](F.gelu(self.correlation_embed[0](difference), approximate='tanh'))
        return self.head(self.backbone(F.gelu(instfreq + difference, approximate='tanh'))).softmax(-1)

    def __call__(self, instfreq, difference):
        probabilities = self.probabilities(instfreq, difference)
        unvoiced = probabilities[:, 0].clone()
        probabilities[:, 0] = -100.0
        band = probabilities.unfold(1, PITCH_BAND, 1).sum(-1)
        top = band.argmax(1)
        top_probability = band.gather(1, top[:, None])[:, 0] + 1e-6
        half = band.gather(1, (top - BINS_PER_OCTAVE).clamp(min=1)[:, None])[:, 0]
        half = torch.where(top <= BINS_PER_OCTAVE, 0.0, half)
        double = band.gather(1, (top + BINS_PER_OCTAVE).clamp(max=PITCH_BINS - PITCH_BAND)[:, None])[:, 0]
        double = torch.where(top > PITCH_BINS - PITCH_BAND - BINS_PER_OCTAVE, 0.0, double)
        mask = (top[:, None] <= self.bins) & (self.bins < top[:, None] + PITCH_BAND)
        quantized = (probabilities * mask).argmax(1)
        return quantized, torch.stack([unvoiced, half / top_probability, double / top_probability])


class BeatriceModel:
    def __init__(self, path, device='cpu', prepend=PHONE_LOOKAHEAD):
        self.device = torch.device(device)
        self.directory, self.name, self.voices = read_paraphernalia(path)
        self.prepend = prepend
        self._load_phone_extractor()
        self._load_pitch_estimator()
        self._load_waveform_generator()
        self._load_speakers()
        self.hann = torch.hann_window(2 * OUT_HOP, device=self.device)
        self.ir_bins = torch.arange(IR_LENGTH // 2 + 1, device=self.device) * (-2.0 * math.pi / IR_LENGTH)
        self.ir_offsets = torch.arange(IR_LENGTH, device=self.device)
        self.pitch_shift = 0.0
        self.vq_neighbors = 0
        self.set_speaker(0)
        self.set_formant_shift(0.0)
        self.reset()

    @property
    def speaker_count(self):
        return self.additive_speakers.shape[0]

    def _load_phone_extractor(self):
        weights = Weights(self.directory / 'phone_extractor.bin', self.device)
        shapes = ((16, 1, 10, 5), (32, 16, 3, 2), (64, 32, 3, 2), (128, 64, 3, 2), (128, 128, 3, 2), (128, 128, 2, 2))
        self.feature_convs = [Conv(weights, inp, out, size, stride, bias=False) for out, inp, size, stride in shapes]
        self.phone_backbone = Stack(weights, 128, 128, 384, 20, 9, 17, attention=PhaseAttention)
        self.phone_head = Conv(weights, 128, PHONE_CHANNELS, 1)
        weights.finish()

    def _load_pitch_estimator(self):
        self.pitch_estimator = PitchEstimator(Weights(self.directory / 'pitch_estimator.bin', self.device), self.device)

    def _load_waveform_generator(self):
        weights = Weights(self.directory / 'waveform_generator.bin', self.device)
        self.embed_phone = Conv(weights, PHONE_CHANNELS, HIDDEN_CHANNELS, 1)
        self.embed_pitch = weights.take(PITCH_BINS, HIDDEN_CHANNELS)
        self.embed_pitch_features = Conv(weights, 4, HIDDEN_CHANNELS, 1)
        self.prenet = Stack(weights, HIDDEN_CHANNELS, HIDDEN_CHANNELS, 512, 4, 7, 33, attention=SpeakerAttention)
        self.ir_generator = Stack(weights, HIDDEN_CHANNELS, HIDDEN_CHANNELS, 512, 2, 3, 33, standardized=True)
        self.ir_post = Conv(weights, HIDDEN_CHANNELS, IR_LENGTH, 1)
        self.ir_window = weights.take(IR_LENGTH)
        self.aperiodicity_generator = Stack(weights, HIDDEN_CHANNELS, HIDDEN_CHANNELS, 512, 1, 3, 33, standardized=True)
        self.aperiodicity_post = Conv(weights, HIDDEN_CHANNELS, OUT_HOP, 1, bias=False)
        self.filter_generator = Stack(weights, HIDDEN_CHANNELS, HIDDEN_CHANNELS, 512, 1, 3, 33, standardized=True)
        self.filter_post = Conv(weights, HIDDEN_CHANNELS, IR_LENGTH, 1, bias=False)
        weights.finish()
        weights = Weights(self.directory / 'embedding_setter.bin', self.device)
        head = KV_CHANNELS // HEADS
        self.speaker_projections = []
        for _ in self.prenet.blocks:
            matrices = [weights.take(2, head, KV_CHANNELS) for _ in range(HEADS)]
            biases = [weights.take(2, head) for _ in range(HEADS)]
            self.speaker_projections.append((torch.stack(matrices), torch.stack(biases)))
        weights.finish()

    def _load_speakers(self):
        path = self.directory / 'speaker_embeddings.bin'
        per_speaker = CODEBOOK_SIZE * PHONE_CHANNELS + HIDDEN_CHANNELS + KV_LENGTH * KV_CHANNELS
        count, remainder = divmod(path.stat().st_size // 2 - FORMANT_STEPS * HIDDEN_CHANNELS, per_speaker)
        if count < 1 or remainder:
            raise ValueError(f'{path.name} does not match the Beatrice {VERSION} layout.')
        weights = Weights(path, self.device)
        self.codebooks = weights.take(count, CODEBOOK_SIZE, PHONE_CHANNELS)
        self.additive_speakers = weights.take(count, HIDDEN_CHANNELS)
        self.formant_embeddings = weights.take(FORMANT_STEPS, HIDDEN_CHANNELS)
        self.speaker_tokens = weights.take(count, KV_LENGTH, KV_CHANNELS)
        weights.finish()

    def set_speaker(self, speaker):
        speaker = int(speaker)
        if not 0 <= speaker < self.speaker_count:
            raise ValueError(f'Speaker must be between 0 and {self.speaker_count - 1}.')
        self.speaker = speaker
        tokens = self.speaker_tokens[speaker]
        for block, (matrices, biases) in zip(self.prenet.blocks, self.speaker_projections):
            projected = torch.einsum('kc,hpdc->phkd', tokens, matrices) + biases.transpose(0, 1)[:, :, None]
            block.attention.keys, block.attention.values = projected.unbind(0)

    def set_formant_shift(self, semitones):
        self.formant_shift = min(max(float(semitones), -2.0), 2.0)
        self.formant_index = int(round(self.formant_shift * 2.0 + 4.0))

    def set_pitch_shift(self, semitones):
        self.pitch_shift = min(max(float(semitones), -24.0), 24.0)

    def set_vq_neighbors(self, count):
        self.vq_neighbors = min(max(int(count), 0), 8)

    def reset(self):
        for module in (self.phone_backbone, self.phone_head, self.pitch_estimator, self.embed_phone,
                       self.embed_pitch_features, self.prenet, self.ir_generator, self.ir_post,
                       self.aperiodicity_generator, self.aperiodicity_post, self.filter_generator, self.filter_post):
            module.reset()
        self.audio = torch.zeros(HISTORY + self.prepend, device=self.device)
        self.frame = 0
        self.phase = torch.rand((), device=self.device, dtype=torch.float64)
        self.last_ir = torch.zeros(1, IR_LENGTH, device=self.device)
        self.periodic_tail = torch.zeros(IR_LENGTH, device=self.device)
        self.excitation = torch.rand(OUT_HOP, device=self.device) - 0.5
        self.noise_tail = torch.zeros(OUT_HOP, device=self.device)
        self.filter_tail = torch.zeros(FILTER_FFT - OUT_HOP, device=self.device)

    def _phones(self, audio):
        x = audio[:, None]
        for conv in self.feature_convs:
            x = F.gelu(conv(x), approximate='tanh')
        phone = self.phone_head(F.gelu(self.phone_backbone(F.layer_norm(x, (x.shape[1],))), approximate='tanh'))
        if self.vq_neighbors:
            codes = self.codebooks[self.speaker]
            nearest = (F.normalize(phone, dim=1, eps=1e-6) @ codes.T).topk(self.vq_neighbors, dim=1).indices
            phone = codes[nearest].mean(1)
        return phone * torch.rsqrt(phone.square().mean(1, keepdim=True) + torch.finfo(torch.float32).eps)

    def _synthesize(self, hertz, ir, aperiodicity, post_filter):
        frames = hertz.shape[0]
        samples = frames * OUT_HOP
        steps = (hertz.double() / OUT_SAMPLE_RATE).repeat_interleave(OUT_HOP)
        phases = self.phase + steps.cumsum(0)
        previous = torch.cat([self.phase.view(1), phases[:-1]])
        marks = (phases.floor() > previous.floor()).nonzero()[:, 0]
        periodic = torch.zeros(samples + IR_LENGTH + 1, device=self.device)
        periodic[1:IR_LENGTH + 1] += self.periodic_tail
        if marks.numel():
            before = previous[marks] - previous[marks].floor()
            after = phases[marks] - phases[marks].floor()
            fraction = ((1.0 - before) / (1.0 - before + after)).float()
            responses = torch.cat([self.last_ir, ir])[torch.div(marks - 1, OUT_HOP, rounding_mode='floor') + 1]
            amplitude = responses[:, :IR_LENGTH // 2 + 1].exp()
            angle = F.pad(responses[:, IR_LENGTH // 2 + 1:], (1, 1))
            angle[:, 1::2] += math.pi
            angle = angle + self.ir_bins * fraction[:, None]
            pulses = torch.fft.irfft(torch.polar(amplitude, angle), n=IR_LENGTH) * self.ir_window
            periodic.index_add_(0, (marks[:, None] + self.ir_offsets).reshape(-1), pulses.reshape(-1))
        self.phase = phases[-1] - phases[-1].floor()
        self.last_ir = ir[-1:]
        self.periodic_tail = periodic[samples + 1:].clone()
        excitation = torch.cat([self.excitation, torch.rand(samples, device=self.device) - 0.5])
        self.excitation = excitation[-OUT_HOP:]
        spectrum = torch.fft.rfft(excitation.unfold(0, 2 * OUT_HOP, OUT_HOP), n=2 * OUT_HOP)
        spectrum[:, 0] = 0.0
        spectrum[:, 1:] *= aperiodicity
        noise = torch.fft.irfft(spectrum, n=2 * OUT_HOP) * self.hann
        aperiodic = torch.zeros(samples + OUT_HOP, device=self.device)
        aperiodic[:samples] += noise[:, :OUT_HOP].reshape(-1)
        aperiodic[OUT_HOP:] += noise[:, OUT_HOP:].reshape(-1)
        aperiodic[:OUT_HOP] += self.noise_tail
        self.noise_tail = aperiodic[samples:].clone()
        signal = (periodic[1:samples + 1] + aperiodic[:samples]).view(frames, OUT_HOP)
        filtered = torch.fft.irfft(
            torch.fft.rfft(signal, n=FILTER_FFT) * torch.fft.rfft(post_filter, n=FILTER_FFT), n=FILTER_FFT
        )
        output = F.fold(filtered.T[None], (1, samples + FILTER_FFT - OUT_HOP), (1, FILTER_FFT),
                        stride=(1, OUT_HOP)).view(-1)
        output[:FILTER_FFT - OUT_HOP] += self.filter_tail
        self.filter_tail = output[samples:].clone()
        return output[:samples]

    @torch.inference_mode()
    def process(self, audio):
        audio = torch.as_tensor(audio, dtype=torch.float32, device=self.device).reshape(-1)
        self.audio = torch.cat([self.audio, audio])
        start = self.frame * IN_HOP - HISTORY
        frames = (start + self.audio.shape[0] - PHONE_LOOKAHEAD - IN_HOP) // IN_HOP - self.frame + 1
        if frames <= 0:
            return torch.zeros(0, device=self.device)
        offset = HISTORY - PHONE_LOOKAHEAD
        phones = self._phones(self.audio[offset:offset + IN_HOP * (frames - 1) + 240])
        instfreq, difference, loudness = self.pitch_estimator.inputs(self.audio, frames)
        quantized, pitch_features = self.pitch_estimator(instfreq, difference)
        quantized = (quantized + round(self.pitch_shift * BINS_PER_OCTAVE / 12)).clamp(1, PITCH_BINS - 1)
        features = torch.cat([loudness[None], pitch_features]).T
        x = (self.embed_phone(phones) + self.embed_pitch[quantized] + self.embed_pitch_features(features)
             + self.additive_speakers[self.speaker] + self.formant_embeddings[self.formant_index])
        x = self.prenet(F.silu(x))
        ir = self.ir_post(F.silu(self.ir_generator(x)))
        aperiodicity = self.aperiodicity_post(F.silu(self.aperiodicity_generator(x)))
        post_filter = self.filter_post(F.silu(self.filter_generator(x)))
        post_filter[:, 0] += 1.0
        hertz = PITCH_BASE_HZ * 2.0 ** (quantized.float() / BINS_PER_OCTAVE)
        self.frame += frames
        self.audio = self.audio[frames * IN_HOP:]
        return self._synthesize(hertz, ir, aperiodicity, post_filter)
