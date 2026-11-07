"""Bounded, local audio inspection for Ilaria RVC."""

import csv
import html
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import zipfile

import numpy as np

from ilaria_dataset_checker import _binary


MAX_FILES = 8
MAX_SECONDS = 180
MAX_SAMPLES = 12_000_000
FORMATS = "wav,flac,mp3,ogg,mov,matroska,aac,aiff,asf,ape,wv,caf,ac3,au"
PLOT_LOCK = threading.Lock()
ANALYSIS_SLOT = threading.BoundedSemaphore(1)
OVERVIEW = ("File", "Result", "Format / codec", "Duration (s)", "Analyzed (s)",
            "Sample rate (Hz)", "Channels", "Peak (dBFS)", "RMS (dBFS)",
            "Silence (%)", "Near full scale (%)")


def _ffmpeg(root, executable, args, timeout):
    result = subprocess.run([_binary(executable, root), *args], capture_output=True,
                            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise ValueError("Could not read the audio. Check its format and try playing the original file.")
    return result.stdout


def _optional_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def _decode(root, path, start, end):
    restrict = ["-protocol_whitelist", "file", "-format_whitelist", FORMATS]
    raw = _ffmpeg(root, "ffprobe", [
        "-v", "error", *restrict, "-select_streams", "a:0", "-show_entries",
        "format=format_name,duration,bit_rate:stream=codec_name,sample_rate,channels,"
        "channel_layout,duration,bit_rate,bits_per_sample,bits_per_raw_sample",
        "-of", "json", "-i", str(path),
    ], 15)
    probe = json.loads(raw)
    streams = probe.get("streams") or []
    if not streams:
        raise ValueError("No audio track found.")
    stream = streams[0]
    container = probe.get("format", {})
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    if not 1000 <= rate <= 384000 or not 1 <= channels <= 8:
        raise ValueError("Supported audio has 1–8 channels and a sample rate up to 384 kHz.")
    duration = _optional_number(stream.get("duration")) or _optional_number(container.get("duration"))
    if duration is not None and start >= duration:
        raise ValueError("Start time is at or after the end of this file.")
    cap = min(MAX_SECONDS, MAX_SAMPLES / (rate * channels))
    requested = (end - start) if end else float("inf")
    available = max(0, duration - start) if duration is not None else float("inf")
    seconds = min(cap, requested, available)
    data = _ffmpeg(root, "ffmpeg", [
        "-nostdin", "-v", "error", *restrict, "-ss", str(start), "-i", str(path),
        "-t", f"{seconds:.9f}", "-map", "0:a:0", "-ar", str(rate), "-ac", str(channels),
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ], 50)
    if not data or len(data) % (4 * channels):
        raise ValueError("No complete audio samples in the requested range.")
    frame_limit = MAX_SAMPLES // channels
    samples = np.frombuffer(data, dtype="<f4").reshape(-1, channels)[:frame_limit]
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains NaN or infinite samples.")
    bitrate = _optional_number(stream.get("bit_rate"))
    depth = _optional_number(stream.get("bits_per_raw_sample")) or _optional_number(stream.get("bits_per_sample"))
    return samples, {
        "container": container.get("format_name", "unknown"),
        "codec": stream.get("codec_name", "unknown"),
        "duration_seconds": duration,
        "audio_bitrate_kbps": bitrate / 1000 if bitrate else None,
        "container_bitrate_kbps": (_optional_number(container.get("bit_rate")) or 0) / 1000 or None,
        "bit_depth": int(depth) if depth else None,
        "sample_rate": rate,
        "channels": channels,
        "channel_layout": stream.get("channel_layout"),
        "start_seconds": start,
        "analyzed_seconds": len(samples) / rate,
        "capped": min(requested, available) > cap and len(samples) / rate >= cap - .1,
    }


def _db(amplitude):
    return round(20 * math.log10(amplitude), 2) + 0.0 if amplitude > 0 else None


def _levels(samples, rate, start, silence_db, min_silence, clip_db):
    channels = samples.shape[1]
    peak = np.max(np.abs(samples), axis=0)
    squares = np.einsum("ij,ij->j", samples, samples, dtype=np.float64)
    dc = samples.mean(axis=0, dtype=np.float64)
    near = np.count_nonzero(np.abs(samples) >= 10 ** (clip_db / 20), axis=0)
    above = int(np.count_nonzero(np.abs(samples) > 1))
    rms = np.sqrt(squares / len(samples))
    overall_rms = float(np.sqrt(squares.sum() / samples.size))
    labels = ("Left", "Right") if channels == 2 else tuple(f"Channel {i + 1}" for i in range(channels))
    per_channel = [[labels[i], _db(float(peak[i])), _db(float(rms[i])),
                    round(float(dc[i]) * 100, 4), round(float(near[i]) * 100 / len(samples), 5)]
                   for i in range(channels)]

    window = max(1, round(rate * .020))
    # Power sums over 20 ms windows; retain the final partial window.
    positions = np.arange(0, len(samples), window)
    counts = np.diff(np.r_[positions, len(samples)])
    frame_power = np.einsum("ij,ij->i", samples, samples, dtype=np.float64) / channels
    powers = np.add.reduceat(frame_power, positions) / counts
    silent = powers <= 10 ** (silence_db / 10)
    edges = np.flatnonzero(np.diff(np.r_[False, silent, False].astype(np.int8)))
    boundaries = np.r_[positions, len(samples)]
    spans = [(boundaries[a] / rate, boundaries[b] / rate) for a, b in zip(edges[::2], edges[1::2])]
    pauses = [[round(start + a, 3), round(start + b, 3), round(b - a, 3)]
              for a, b in spans if b - a + 1e-9 >= min_silence]
    rms_db = _db(overall_rms)
    peak_db = _db(float(peak.max()))
    active_power = float(np.dot(powers[~silent], counts[~silent]) / counts[~silent].sum()) if np.any(~silent) else 0
    metrics = {
        "peak_dbfs": peak_db, "rms_dbfs": rms_db, "active_rms_dbfs": _db(math.sqrt(active_power)),
        "crest_db": round(peak_db - rms_db, 2) if rms_db is not None else None,
        "silence_percent": round(float(counts[silent].sum()) * 100 / len(samples), 2),
        "longest_silence_seconds": round(max((b - a for a, b in spans), default=0), 3),
        "near_full_scale_percent": round(float(near.sum()) * 100 / samples.size, 5),
        "above_full_scale_samples": above,
        "stereo_correlation": None, "left_minus_right_db": None,
    }
    if channels == 2:
        variance = np.maximum(0, squares - len(samples) * dc ** 2)
        if float(variance.min()) > 1e-12:
            cross = np.einsum("i,i->", samples[:, 0], samples[:, 1], dtype=np.float64)
            metrics["stereo_correlation"] = round(float(np.clip(
                (cross - len(samples) * dc.prod()) / math.sqrt(float(variance.prod())), -1, 1)), 3)
        if all(value > 0 for value in rms):
            metrics["left_minus_right_db"] = _db(float(rms[0] / rms[1]))
    return metrics, per_channel, pauses


def _channel(samples, mode):
    if mode == "Left":
        return samples[:, 0]
    if mode in ("Right", "Side"):
        if samples.shape[1] != 2:
            raise ValueError(f"{mode} view needs a two-channel recording. Choose Mix or Left.")
        if mode == "Right":
            return samples[:, 1]
        return (samples[:, 0] - samples[:, 1]) / 2
    return samples.mean(axis=1, dtype=np.float32)


def _spectral(mono, rate, n_fft, scale):
    from scipy.fft import rfft, rfftfreq

    padded = np.pad(mono, (n_fft // 2, n_fft // 2))
    count = min(900, max(2, 1 + len(mono) // (n_fft // 4)))
    starts = np.linspace(0, max(0, len(mono) - 1), count, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[starts]
    hann = np.hanning(n_fft).astype(np.float32)
    amplitude = np.abs(rfft(windows * hann, axis=1)).T * (2 / hann.sum())
    amplitude[[0, -1]] *= .5
    power = amplitude ** 2
    hz = rfftfreq(n_fft, 1 / rate)
    average = power.mean(axis=1, dtype=np.float64)
    cumulative = average.cumsum()
    bandwidth = float(hz[np.searchsorted(cumulative, .95 * cumulative[-1])]) if cumulative[-1] > 1e-20 else None
    if scale == "Mel":
        from librosa.filters import mel
        from librosa import mel_frequencies

        filters = mel(sr=rate, n_fft=n_fft, n_mels=80, norm=None)
        power = filters @ power
        hz_view = mel_frequencies(n_mels=82, fmax=rate / 2)[1:-1]
    elif scale == "Log":
        power, hz_view = power[1:], hz[1:]
    else:
        hz_view = hz
    return starts / rate, hz_view, power, hz, average, bandwidth


def _pitch(mono, rate, lo, hi):
    from scipy.signal import resample_poly
    import pyworld

    gcd = math.gcd(rate, 16000)
    resampled = resample_poly(mono[:rate * 60], 16000 // gcd, rate // gcd)
    values = np.ascontiguousarray(resampled, dtype=np.float64)
    if len(values) < 1600:
        return np.empty(0), np.empty(0)
    f0, times = pyworld.dio(values, 16000, f0_floor=lo, f0_ceil=hi, frame_period=10)
    return times, pyworld.stonemask(values, f0, times, 16000)


def _plot(path, mono, meta, settings):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rate, start = meta["sample_rate"], meta["start_seconds"]
    times, hz_view, power, hz, spectrum, bandwidth = _spectral(
        mono, rate, settings["n_fft"], settings["scale"])
    time_end = start + len(mono) / rate
    pitch_info = None
    if settings["show_pitch"]:
        f0_t, f0 = _pitch(mono, rate, settings["pitch_min"], settings["pitch_max"])
        voiced = f0[f0 > 0]
        pitch_info = {
            "analyzed_seconds": round(min(60, len(mono) / rate), 3),
            "voiced_percent": round(100 * len(voiced) / len(f0), 2) if len(f0) else 0,
            "median_hz": round(float(np.median(voiced)), 1) if len(voiced) else None,
        }
    with PLOT_LOCK:
        figure = Figure(figsize=(11, 8.3 if pitch_info is None else 10), layout="constrained", facecolor="#36243c")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3 if pitch_info is None else 4, 1)
        for ax in axes:
            ax.set_facecolor("#251a2b")
            ax.tick_params(colors="#f6daeb")
            for spine in ax.spines.values():
                spine.set_color("#a775a9")
            ax.xaxis.label.set_color("#f6daeb")
            ax.yaxis.label.set_color("#f6daeb")
            ax.title.set_color("#ffd5ea")
        wave_ax, spec_ax, spectrum_ax = axes[:3]
        edges = np.linspace(0, len(mono), min(1600, len(mono)) + 1, dtype=int)
        wave_ax.fill_between(start + edges[:-1] / rate,
                             np.minimum.reduceat(mono, edges[:-1]),
                             np.maximum.reduceat(mono, edges[:-1]), color="#ffc0cb")
        wave_ax.set(title=f"Waveform · {settings['channel']} view", ylabel="Amplitude", xlim=(start, time_end))
        wave_ax.axhline(0, color="#b386ae", linewidth=.5)
        with np.errstate(divide="ignore"):
            db = 10 * np.log10(np.maximum(power, 1e-15))
        upper = max(-120, float(db.max()))
        plot = spec_ax.pcolormesh(start + times, hz_view, db, shading="auto", cmap="magma",
                                  vmin=upper - settings["db_range"], vmax=upper, rasterized=True)
        if settings["scale"] == "Log" and len(hz_view) > 1:
            spec_ax.set_yscale("log")
            # The first positive FFT bin sets the lowest meaningful displayed frequency.
            lower = max(20, float(hz[1]))
            upper_hz = rate / 2
            spec_ax.set_ylim(lower, upper_hz)
            ticks = [f for f in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
                     if lower <= f <= upper_hz]
            spec_ax.set_yticks(ticks, [f"{f:,}" for f in ticks])
        spec_ax.set(title=f"Spectrogram · {settings['scale']} frequency", xlabel="Time (s)",
                    ylabel="Frequency (Hz)", xlim=(start, time_end))
        bar = figure.colorbar(plot, ax=spec_ax, pad=.015)
        bar.ax.tick_params(colors="#f6daeb")
        bar.set_label("FFT level (dB, relative scale)", color="#f6daeb")
        if len(hz) > 2:
            spectrum_ax.semilogx(hz[1:], 10 * np.log10(np.maximum(spectrum[1:], 1e-15)), color="#ffc0cb")
        spectrum_ax.set(title="Average spectrum", xlabel="Frequency (Hz)", ylabel="Power (dB)")
        spectrum_ax.grid(alpha=.17)
        if pitch_info is not None:
            axes[3].scatter(start + f0_t[f0 > 0], f0[f0 > 0], s=4, color="#ffc0cb")
            axes[3].set(title="Estimated pitch · up to 60 s", xlabel="Time (s)", ylabel="Pitch (Hz)",
                        xlim=(start, min(time_end, start + 60)))
        figure.savefig(path, dpi=120, facecolor=figure.get_facecolor())
        figure.clear()
    return bandwidth, pitch_info


def _fmt(value, suffix=""):
    return f"{value:g}{suffix}" if value is not None else "—"


def _details(entry, settings, bandwidth, pitch_info):
    m, q = entry["metadata"], entry["measurements"]
    filename = html.escape(entry["file"]).replace("`", "&#96;").replace("\n", " ")
    pitch_text = ""
    if pitch_info is not None:
        pitch_text = (f"\n\n**Pitch estimate**  {_fmt(pitch_info['median_hz'], ' Hz')} median · "
                      f"{_fmt(pitch_info['voiced_percent'], '%')} voiced · "
                      f"first {_fmt(pitch_info['analyzed_seconds'], ' s')} of the range.")
    notes = []
    if m["capped"]:
        notes.append("The sample limit shortened this range. Move the start time to inspect a later section.")
    if q["above_full_scale_samples"]:
        notes.append(f"{q['above_full_scale_samples']:,} samples exceed digital full scale.")
    if q["stereo_correlation"] is not None and q["stereo_correlation"] < -.2:
        notes.append("Negative stereo correlation. Listen in mono to check for cancellation.")
    notes.append("Near full scale is a screening measure. It does not establish that a recording was clipped.")
    return (
        f"### Detail · `{filename}`\n\n"
        f"**Source**  {m['container']} / {m['codec']} · {m['sample_rate']:,} Hz · "
        f"{m['channels']} ch · {_fmt(m['bit_depth'], ' bit')} · "
        f"{_fmt(m['audio_bitrate_kbps'], ' kb/s audio')} · {_fmt(m['container_bitrate_kbps'], ' kb/s container')}\n\n"
        f"**Analyzed range**  {_fmt(m['start_seconds'], ' s')} to "
        f"{_fmt(m['start_seconds'] + m['analyzed_seconds'], ' s')}\n\n"
        f"**Levels**  Peak {_fmt(q['peak_dbfs'], ' dBFS')} · RMS {_fmt(q['rms_dbfs'], ' dBFS')} · "
        f"Active RMS {_fmt(q['active_rms_dbfs'], ' dBFS')} · Crest {_fmt(q['crest_db'], ' dB')}\n\n"
        f"**Silence**  {q['silence_percent']:g}% below {settings['silence_db']:g} dBFS · "
        f"longest pause {_fmt(q['longest_silence_seconds'], ' s')} · {len(entry['silence_ranges'])} marked ranges\n\n"
        f"**Stereo**  Correlation {_fmt(q['stereo_correlation'])} · L−R RMS {_fmt(q['left_minus_right_db'], ' dB')}\n\n"
        f"**Spectral bandwidth**  {_fmt(bandwidth, ' Hz')} contains approximately 95% of measured FFT energy."
        + pitch_text + "\n\n" + "\n".join("- " + note for note in notes)
    )


def _export(directory, export_format, report, rows, plot):
    if export_format == "None":
        return None
    if export_format in ("JSON", "ZIP"):
        js = directory / "audio-analysis.json"
        js.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        if export_format == "JSON":
            return str(js)
    csv_path = directory / "audio-analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(OVERVIEW)
        for row in rows:
            writer.writerow(["'" + x if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@"))
                             else x for x in row])
    if export_format == "CSV":
        return str(csv_path)
    archive = directory / "audio-analysis.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as file:
        for item in (js, csv_path, plot):
            if item:
                file.write(item, item.name)
    return str(archive)


def analyze_uploads(root, uploads, start, end, silence_db, min_silence, clip_db,
                    channel, scale, n_fft, db_range, show_pitch, pitch_min, pitch_max,
                    export_format):
    """Yield progress and analysis data. One detailed visual per upload group."""
    if not ANALYSIS_SLOT.acquire(blocking=False):
        yield "An analysis is already running. Wait for it to finish.", [], "", [], [], None, None, None
        return
    try:
        yield from _analyze_uploads(root, uploads, start, end, silence_db, min_silence, clip_db,
                                    channel, scale, n_fft, db_range, show_pitch, pitch_min,
                                    pitch_max, export_format)
    finally:
        ANALYSIS_SLOT.release()


def _analyze_uploads(root, uploads, start, end, silence_db, min_silence, clip_db,
                     channel, scale, n_fft, db_range, show_pitch, pitch_min, pitch_max,
                     export_format):
    rows, results = [], []
    detail, channel_rows, silence_rows, plot, preview = "", [], [], None, None
    if not uploads or len(uploads) > MAX_FILES:
        yield f"Upload 1–{MAX_FILES} audio files.", rows, detail, channel_rows, silence_rows, None, None, None
        return
    try:
        start, end, silence_db, min_silence, clip_db, db_range, pitch_min, pitch_max = (
            float(value) for value in (start, end, silence_db, min_silence, clip_db, db_range, pitch_min, pitch_max))
        n_fft = int(n_fft)
        if not (math.isfinite(start) and math.isfinite(end) and start >= 0 and (end == 0 or end > start)):
            raise ValueError("Enter an end time greater than the start time, or use 0 for automatic end.")
        if not (-90 <= silence_db <= 0 and .02 <= min_silence <= 180 and -12 <= clip_db <= 0 and 20 <= db_range <= 120):
            raise ValueError("Check the silence, near-full-scale and spectrogram thresholds.")
        if not 40 <= pitch_min < pitch_max <= 1200:
            raise ValueError("Pitch maximum must exceed minimum. Supported range: 40–1200 Hz.")
        if (n_fft not in (512, 1024, 2048, 4096) or scale not in ("Linear", "Log", "Mel")
                or channel not in ("Mix", "Left", "Right", "Side")
                or export_format not in ("None", "CSV", "JSON", "ZIP")):
            raise ValueError("Choose valid view and export options.")
    except (ValueError, TypeError, OverflowError) as error:
        yield str(error), rows, detail, channel_rows, silence_rows, None, None, None
        return
    settings = dict(start_seconds=start, end_seconds=end, silence_db=silence_db, min_silence=min_silence,
                    near_full_scale_db=clip_db, channel=channel, scale=scale, n_fft=n_fft,
                    db_range=db_range, show_pitch=bool(show_pitch), pitch_min=pitch_min, pitch_max=pitch_max)
    directory = Path(tempfile.mkdtemp(prefix="audio-analyzer-", dir=Path(root) / "TEMP"))
    for idx, upload in enumerate(uploads, 1):
        path = Path(upload.name).resolve()
        name = Path(getattr(upload, "orig_name", None) or path.name).name
        yield (f"Analyzing {idx}/{len(uploads)} · {name}", list(rows), detail, channel_rows,
               silence_rows, str(plot) if plot else None, preview, None)
        try:
            samples, meta = _decode(root, path, start, end)
            stats, channels, pauses = _levels(samples, meta["sample_rate"], start, silence_db, min_silence, clip_db)
            entry = {"file": name, "metadata": meta, "measurements": stats,
                     "channels": channels, "silence_ranges": pauses}
            if plot is None:
                try:
                    import soundfile

                    mono = _channel(samples, channel)
                    candidate = directory / "audio-analysis.png"
                    bandwidth, pitch_info = _plot(candidate, mono, meta, settings)
                    preview_path = directory / "analysis-preview.wav"
                    sample_rate = meta["sample_rate"]
                    soundfile.write(str(preview_path), mono[:sample_rate * 30], sample_rate, subtype="PCM_16")
                    entry.update(spectral_bandwidth_95_hz=bandwidth, pitch=pitch_info)
                    detail = _details(entry, settings, bandwidth, pitch_info)
                    channel_rows, silence_rows, plot, preview = channels, pauses[:300], candidate, str(preview_path)
                except (RuntimeError, ValueError, OSError) as error:
                    entry["view_error"] = str(error)
                    detail = f"Measurements are available. The selected view could not be made: {html.escape(str(error))}"
            results.append(entry)
            rows.append([name, "Range capped" if meta["capped"] else "OK",
                         f"{meta['container']} / {meta['codec']}",
                         round(meta["duration_seconds"], 3) if meta["duration_seconds"] is not None else None,
                         round(meta["analyzed_seconds"], 3), meta["sample_rate"], meta["channels"],
                         stats["peak_dbfs"], stats["rms_dbfs"], stats["silence_percent"],
                         stats["near_full_scale_percent"]])
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            message = "Decoder timed out." if isinstance(error, subprocess.TimeoutExpired) else str(error)
            results.append({"file": name, "error": message})
            rows.append([name, message, *([None] * 9)])
        finally:
            # Release the full decoded buffer before opening the next upload.
            samples = mono = None
    succeeded = sum("metadata" in entry for entry in results)
    status = f"Analyzed {succeeded}/{len(uploads)} files."
    if len(silence_rows) == 300:
        status += " Showing up to 300 silence ranges. The JSON report includes all ranges."
    try:
        export = _export(directory, export_format, {"settings": settings, "files": results}, rows, plot)
    except (OSError, ValueError) as error:
        export = None
        status += " Could not create report: " + str(error)
    yield status, rows, detail, channel_rows, silence_rows, str(plot) if plot else None, preview, export
