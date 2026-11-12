"""Optional input formant shaping, noise reduction and output effects."""

import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


POST_EFFECTS = (
    "Reverb", "Pitch Shift", "Limiter", "Gain", "Distortion", "Chorus",
    "Bitcrush", "Clipping", "Compressor", "Delay",
)
FFMPEG = Path(__file__).resolve().parents[1] / "ffmpeg.exe"


def apply_formant_shift(audio, sample_rate, ratio):
    """Move the formants while retaining the original pitch and duration."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0.7 <= ratio <= 1.4:
        raise ValueError("Formant ratio must be between 0.7 and 1.4.")
    if ratio == 1.0 or len(audio) < sample_rate / 15 or not np.any(audio):
        return audio

    import parselmouth

    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    shifted = parselmouth.praat.call(
        sound, "Change gender", 50.0, 1000.0, ratio, 0.0, 1.0, 1.0
    )
    if round(shifted.sampling_frequency) != sample_rate:
        shifted = shifted.resample(sample_rate)
    values = np.asarray(shifted.values[0], dtype=np.float32)
    if len(values) < len(audio):
        values = np.pad(values, (0, len(audio) - len(values)))
    return values[:len(audio)]


def _float_audio(audio):
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.signedinteger):
        return audio.astype(np.float32) / float(-np.iinfo(audio.dtype).min)
    return audio.astype(np.float32, copy=False)


def _filter_audio(audio, sample_rate, filters):
    executable = str(FFMPEG) if FFMPEG.is_file() else shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("Post-Process needs the FFmpeg executable.")
    raw_audio = audio.astype("<f4", copy=False)
    completed = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
         "-af", ",".join(filters), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
        input=raw_audio.tobytes(), capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not apply the selected effects. {detail[:400]}".strip())
    return np.frombuffer(completed.stdout, dtype="<f4")


def _level(db):
    db = float(db)
    if not math.isfinite(db) or not -60 <= db <= 60:
        raise ValueError("The level must be between -60 and 60 dB.")
    return 10 ** (db / 20)


def _limiter_filters(sample_rate, threshold_db):
    threshold_db = float(threshold_db)
    if not math.isfinite(threshold_db) or not -60 <= threshold_db <= 0:
        raise ValueError("Limiter threshold must be between -60 and 0 dB.")
    # The bundled limiter cannot accept thresholds below 1/16 linear.
    # Scale the signal around the limiter to support quieter thresholds.
    threshold = _level(threshold_db)
    floor = 0.0625
    scale = max(1.0, floor / threshold)
    lookahead = max(1, int(sample_rate * 0.005))
    return [
        f"volume={scale:.10f}",
        f"apad=pad_len={lookahead}",
        f"alimiter=limit={max(floor, threshold):.10f}:attack=5:release=50:level=0",
        f"atrim=start_sample={lookahead}",
        f"volume={1 / scale:.10f}",
    ]


def apply_output_effects(audio, sample_rate, options):
    """Apply enabled effects to the converted mono audio."""
    clean_audio = options.get("clean_audio", False)
    effects = (options.get("effects") or ()) if options.get("post_process") else ()
    if len(audio) == 0 or not (clean_audio or effects):
        return audio

    audio = _float_audio(audio)
    if clean_audio and np.any(audio):
        strength = float(options.get("clean_strength", 0.5))
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Clean Strength must be between 0 and 1.")
        if strength and len(audio) >= 128:
            from noisereduce import reduce_noise

            audio = reduce_noise(y=audio, sr=sample_rate, prop_decrease=strength)

    if "Reverb" in effects:
        audio = _filter_audio(audio, sample_rate, [
            "aecho=in_gain=0.8:out_gain=0.8:delays=23|41|67|89|131:decays=0.28|0.2|0.13|0.08|0.04",
        ])
    if "Pitch Shift" in effects:
        steps = float(options.get("post_pitch", 0))
        if not math.isfinite(steps) or not -12 <= steps <= 12:
            raise ValueError("Output pitch must be between -12 and 12 semitones.")
        if steps and len(audio) > 1:
            import librosa

            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)

    filters = []
    if "Limiter" in effects:
        filters.extend(_limiter_filters(sample_rate, options.get("limiter_db", -6)))
    if "Gain" in effects:
        filters.append(f"volume={_level(options.get('gain_db', 0)):.10f}")
    if "Distortion" in effects:
        filters.extend(("volume=17.782794", "asoftclip=type=tanh"))
    if "Chorus" in effects:
        filters.append("chorus=in_gain=0.7:out_gain=0.8:delays=7:decays=0.5:speeds=1:depths=0.25")
    if "Bitcrush" in effects:
        filters.append("aeval=exprs='round(val(0)*128)/128'")
    if "Clipping" in effects:
        filters.append("aeval=exprs='clip(val(0),-0.501187,0.501187)'")
    if "Compressor" in effects:
        filters.append("acompressor=threshold=0.12589254:ratio=3:attack=1:release=100")
    if "Delay" in effects:
        filters.append("aecho=in_gain=1:out_gain=0.5:delays=500:decays=1")
    if filters:
        audio = _filter_audio(audio, sample_rate, filters)
    return np.clip(audio, -1, 1).astype(np.float32, copy=False)
