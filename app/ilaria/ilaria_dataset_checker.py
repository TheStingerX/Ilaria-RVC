"""Read-only, bounded-memory checks for RVC source recordings.

The numbers are screening measurements, not a judgement of voice quality.
No source audio is normalized, rewritten, deleted or copied by this module.
"""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = frozenset((
    ".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a",
    ".aac", ".wma", ".aif", ".aiff", ".aifc", ".webm", ".mp4", ".mkv",
))
_SOUNDFILE_ERRORS = (RuntimeError, sf.LibsndfileError) if hasattr(sf, "LibsndfileError") else (RuntimeError,)


def _db(value):
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def _db_text(value):
    return "silent" if value is None or not math.isfinite(value) else "%.1f dBFS" % value


def _duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)


def _cell(value):
    value = html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()!])", r"\\\1", value)


class _Measurements:
    def __init__(self, sample_rate, channels, silence_db, min_silence_seconds):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        if not 1000 <= self.sample_rate <= 768000 or not 1 <= self.channels <= 64:
            raise ValueError("Unsupported sample rate or channel count")
        self.window = max(1, int(round(self.sample_rate * 0.020)))
        self.pending = np.empty((0, self.channels), dtype=np.float32)
        self.threshold = 10.0 ** (silence_db / 10.0)
        self.min_silence_frames = min_silence_seconds * self.sample_rate
        self.frames = 0
        self.peak = 0.0
        self.clipped_samples = 0
        self.over_full_scale = 0
        self.clip_run = 0
        self.longest_clip_run = 0
        self.silent_run = 0
        self.longest_silence = 0
        self.longest_silence_start = 0
        self.analyzed_frames = 0
        self.long_silence_count = 0
        self.silent_frames = 0
        self.active_squares = 0.0
        self.active_samples = 0

    def _windows(self, data, final=False):
        if not len(data):
            return
        squared = np.square(data, dtype=np.float64)
        for start in range(0, len(data), self.window):
            block = squared[start:start + self.window]
            count = len(block)
            if not final and count != self.window:
                raise AssertionError("Incomplete analysis window")
            # Energy across channels avoids cancelling opposite-phase stereo.
            energy = float(block.mean())
            if energy <= self.threshold:
                self.silent_run += count
                self.silent_frames += count
                if self.silent_run > self.longest_silence:
                    self.longest_silence = self.silent_run
                    self.longest_silence_start = self.analyzed_frames + count - self.silent_run
            else:
                if self.silent_run >= self.min_silence_frames:
                    self.long_silence_count += 1
                self.silent_run = 0
                self.active_squares += float(block.sum())
                self.active_samples += block.size
            self.analyzed_frames += count

    def add(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError("Decoder returned an unexpected channel layout")
        if not np.isfinite(data).all():
            raise ValueError("Audio contains invalid (NaN or infinite) samples")
        if not len(data):
            return
        absolute = np.abs(data)
        self.peak = max(self.peak, float(absolute.max()))
        near_full = absolute >= 0.999
        self.clipped_samples += int(near_full.sum())
        self.over_full_scale += int((absolute > 1.0).sum())
        frame_clipped = np.any(near_full, axis=1)
        # Run lengths, including runs crossing decoder-block boundaries.
        changes = np.flatnonzero(np.diff(np.r_[False, frame_clipped, False].astype(np.int8)))
        if len(changes):
            starts, ends = changes[::2], changes[1::2]
            lengths = ends - starts
            if starts[0] == 0:
                lengths[0] += self.clip_run
            self.longest_clip_run = max(self.longest_clip_run, int(lengths.max()))
            self.clip_run = int(lengths[-1]) if ends[-1] == len(data) else 0
        else:
            self.clip_run = 0
        self.frames += len(data)
        data = np.concatenate((self.pending, data), axis=0)
        complete = len(data) // self.window * self.window
        self._windows(data[:complete])
        # Copy the remainder so a small view does not retain the whole block.
        self.pending = data[complete:].copy()

    def finish(self):
        self._windows(self.pending, final=True)
        self.pending = np.empty((0, self.channels), dtype=np.float32)
        if self.silent_run >= self.min_silence_frames:
            self.long_silence_count += 1
        if not self.frames:
            raise ValueError("Audio has no decoded samples")
        rms = math.sqrt(self.active_squares / self.active_samples) if self.active_samples else 0.0
        clip_ratio = self.clipped_samples / (self.frames * self.channels)
        return {
            "duration_seconds": self.frames / self.sample_rate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "peak_dbfs": _db(self.peak),
            "active_rms_dbfs": _db(rms),
            "longest_silence_start": self.longest_silence_start / self.sample_rate,
            "longest_silence_seconds": self.longest_silence / self.sample_rate,
            "long_silence_count": self.long_silence_count,
            "silence_fraction": self.silent_frames / self.frames,
            "clipped_fraction": clip_ratio,
            "clipped_samples": self.clipped_samples,
            "longest_clipped_frames": self.longest_clip_run,
            "clipping_suspected": self.longest_clip_run >= 3 or (clip_ratio >= 0.0001 and self.clipped_samples >= 3),
            "above_full_scale_samples": self.over_full_scale,
            "silent": self.active_samples == 0,
        }


def _binary(name, root):
    root = Path(root or (Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "ilaria" else Path(__file__).resolve().parent))
    for candidate in (root / (name + ".exe"), root / name):
        if candidate.is_file():
            return str(candidate)
    located = shutil.which(name)
    if not located:
        raise ValueError("%s is missing; keep the bundled decoder in the application folder" % name)
    return located


def _hidden_process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _soundfile_measure(path, silence_db, min_silence_seconds):
    # libsndfile can silently accept a truncated PCM WAV and shorten its frame
    # count. Check the declared RIFF length before accepting that decode.
    with open(path, "rb") as header_stream:
        header = header_stream.read(12)
    if len(header) == 12 and header[:4] in (b"RIFF", b"RIFX") and header[8:] == b"WAVE":
        endian = "little" if header[:4] == b"RIFF" else "big"
        declared_size = int.from_bytes(header[4:8], endian) + 8
        if declared_size > path.stat().st_size:
            raise ValueError("Truncated WAV container: the recording is shorter than its header declares")
    with sf.SoundFile(str(path), "r") as audio:
        expected_frames = len(audio)
        stats = _Measurements(audio.samplerate, audio.channels, silence_db, min_silence_seconds)
        # At most 65,536 float samples per block, irrespective of channel count.
        frames_per_block = max(1, 65536 // audio.channels)
        while True:
            block = audio.read(frames_per_block, dtype="float32", always_2d=True)
            if not len(block):
                break
            stats.add(block)
        if stats.frames < expected_frames:
            raise ValueError("Audio ended before its declared length (possibly damaged)")
        return stats.finish()


def _ffmpeg_measure(path, silence_db, min_silence_seconds, root):
    probe = subprocess.run([
        _binary("ffprobe", root), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
        "-of", "json", str(path),
    ], capture_output=True, timeout=30, **_hidden_process_options())
    if probe.returncode:
        raise ValueError("Audio could not be decoded: " + probe.stderr.decode("utf-8", "replace")[-300:].strip())
    metadata = json.loads(probe.stdout.decode("utf-8"))["streams"]
    if not metadata:
        raise ValueError("The file contains no audio track")
    sample_rate = int(metadata[0]["sample_rate"])
    channels = int(metadata[0]["channels"])
    stats = _Measurements(sample_rate, channels, silence_db, min_silence_seconds)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            _binary("ffmpeg", root), "-nostdin", "-v", "error", "-xerror",
            "-err_detect", "explode", "-protocol_whitelist", "file,pipe", "-i", str(path),
            "-map", "0:a:0", "-vn", "-sn", "-dn", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=errors, **_hidden_process_options())
        try:
            remainder = b""
            while True:
                block = process.stdout.read(262144)
                if not block:
                    break
                block = remainder + block
                aligned = len(block) // (4 * channels) * (4 * channels)
                if aligned:
                    stats.add(np.frombuffer(block[:aligned], dtype="<f4").reshape(-1, channels))
                remainder = block[aligned:]
            code = process.wait(timeout=30)
            if code or remainder:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 600))
                detail = errors.read().decode("utf-8", "replace").strip()
                raise ValueError("Audio decoder reported a damaged or unsupported file" + (": " + detail[-300:] if detail else ""))
            return stats.finish()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def analyze_file(path, silence_db=-45.0, min_silence_seconds=2.0, root=None):
    """Return measurements or a readable per-file error; never modify audio."""
    path = Path(path).resolve()
    try:
        # Do not allow a decoder fallback to hide invalid PCM data.
        try:
            result = _soundfile_measure(path, silence_db, min_silence_seconds)
            result["decoder"] = "soundfile"
        except _SOUNDFILE_ERRORS:
            result = _ffmpeg_measure(path, silence_db, min_silence_seconds, root)
            result["decoder"] = "ffmpeg"
        result.update({"name": path.name, "error": None, "warnings": []})
        if result["silent"]:
            result["warnings"].append("No audio above the silence threshold")
        elif result["long_silence_count"]:
            result["warnings"].append("%s long silent gap(s)" % result["long_silence_count"])
        if result["clipping_suspected"]:
            result["warnings"].append("Possible clipping")
        if result["above_full_scale_samples"]:
            result["warnings"].append("Samples exceed full scale")
        return result
    except Exception as error:
        return {"name": path.name, "error": str(error), "warnings": ["Cannot safely read audio"]}


def _report(results, silence_db, min_silence_seconds, volume_tolerance_db, ignored):
    good = [item for item in results if not item["error"]]
    active = [item for item in good if not item["silent"]]
    median = float(np.median([item["active_rms_dbfs"] for item in active])) if active else None
    for item in good:
        # Rebuild the cross-file warning so repeated progress rendering is stable.
        item["warnings"] = [warning for warning in item["warnings"] if not warning.startswith("Volume ")]
        if median is not None and not item["silent"] and abs(item["active_rms_dbfs"] - median) >= volume_tolerance_db:
            delta = item["active_rms_dbfs"] - median
            item["warnings"].append("Volume %.1f dB %s dataset median" % (abs(delta), "above" if delta > 0 else "below"))
    flagged = sum(bool(item["warnings"]) for item in results)
    damaged = len(results) - len(good)
    lines = [
        "**%s checked | %s file(s) to review | %s unreadable | %s readable audio**" % (
            len(results), flagged, damaged, _duration(sum(item["duration_seconds"] for item in good))),
        "",
        "Source files are unchanged. Checks cover files directly inside the selected folder; subfolders are not scanned.",
        "",
        "| File | Duration | Peak | Active volume | Longest silence | Near full scale | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        if item["error"]:
            lines.append("| %s | - | - | - | - | - | **Unreadable:** %s |" % (_cell(item["name"]), _cell(item["error"])))
        else:
            lines.append("| %s | %s | %s | %s | %.2f s | %.3f%% | %s |" % (
                _cell(item["name"]), _duration(item["duration_seconds"]), _db_text(item["peak_dbfs"]),
                _db_text(item["active_rms_dbfs"]), item["longest_silence_seconds"],
                item["clipped_fraction"] * 100.0, _cell("; ".join(item["warnings"]) or "No issues detected")))
    lines.extend([
        "", "**How to read this**",
        "",
        "- **Silence:** at least %.1f seconds below %.1f dBFS, measured in 20 ms windows. Quiet breaths or very quiet speech may also be flagged; listen before cutting anything." % (min_silence_seconds, silence_db),
        "- **Possible clipping:** samples near full scale (at least 0.999) appear in a run of three or more frames, or affect at least 0.01% of samples (three samples minimum). A single peak is not enough. Inspect the waveform; lowering volume does not repair existing distortion.",
        "- **Active volume:** RMS level excluding windows below the silence threshold. A difference of %.1f dB or more from the dataset median is flagged. Listen and adjust unusually loud or quiet recordings before training; these are screening levels, not perceptual LUFS." % volume_tolerance_db,
        "- **Unreadable:** the decoder could not safely read the recording. Re-export it from the original source, then check again. A successful decode cannot prove that every recording is free of corruption or audible defects.",
        "- Keep useful breaths, laughs and vocal expression. These checks do not identify unwanted speakers, music, noise or poor pronunciation.",
    ])
    rates = sorted(set(item["sample_rate"] for item in good))
    if rates:
        lines.extend(["", "Source sample rates: %s Hz. Different source rates are allowed; Process Data resamples them to the training rate." % ", ".join(map(str, rates))])
    if ignored:
        lines.extend(["", "%s unsupported file(s) or subfolder(s) skipped." % ignored])
    return "\n".join(lines)


def check_dataset(dataset_path, silence_db=-45.0, min_silence_seconds=2.0, volume_tolerance_db=6.0, root=None, on_complete=None):
    """Yield (one-line percentage status, Markdown report) for Gradio."""
    try:
        silence_db = float(silence_db)
        min_silence_seconds = float(min_silence_seconds)
        volume_tolerance_db = float(volume_tolerance_db)
        if not (-100 <= silence_db < 0 and 0.1 <= min_silence_seconds <= 60 and 0.1 <= volume_tolerance_db <= 60):
            raise ValueError("Use silence -100 to below 0 dBFS, gaps 0.1 to 60 seconds, and volume difference 0.1 to 60 dB")
        folder = Path(str(dataset_path or "").strip().strip('"')).expanduser()
        if not str(dataset_path or "").strip():
            raise ValueError("Enter the Path to Dataset in Training Settings first")
        if not folder.is_absolute():
            folder = Path(root or (Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "ilaria" else Path(__file__).resolve().parent)) / folder
        if not folder.is_dir():
            raise ValueError("Dataset folder was not found: " + str(folder))
        entries = list(folder.iterdir())
        files = sorted((path for path in entries if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS), key=lambda path: path.name.casefold())
        ignored = len(entries) - len(files)
        if not files:
            raise ValueError("No supported audio files were found directly in this folder. Move audio out of subfolders or choose the correct folder")
        yield "0.0%% | Checking dataset: %s audio file(s)" % len(files), "Checking recordings. Source files are unchanged."
        results = []
        for index, path in enumerate(files, 1):
            yield "%.1f%% | Reading %s (%s/%s)" % ((index - 1) / len(files) * 100.0, path.name, index, len(files)), "Checked %s of %s recordings. Full results appear when the check finishes." % (index - 1, len(files))
            results.append(analyze_file(path, silence_db, min_silence_seconds, root))
            # Publish completion per file; never present 100% before report is ready.
            if index < len(files):
                yield "%.1f%% | Checked %s (%s/%s)" % (index / len(files) * 100.0, path.name, index, len(files)), "Checked %s of %s recordings. Full results appear when the check finishes." % (index, len(files))
        report = _report(results, silence_db, min_silence_seconds, volume_tolerance_db, ignored)
        if on_complete is not None:
            on_complete(folder, results)
        flagged = sum(bool(item["warnings"]) for item in results)
        yield "100.0%% | Dataset check done! %s file(s) checked; %s to review" % (len(files), flagged), report
    except Exception as error:
        yield "Dataset check stopped: " + str(error), "No source files have been changed."

    
def preview_recording(root, scan, name, offset, around_silence):
    if not scan or not name or name not in scan["results"]:
        return "Run Check dataset and choose a recording first.", None
    folder = Path(scan["folder"])
    path = folder / name
    if path.resolve().parent != folder.resolve() or not path.is_file():
        return "Recording unavailable.", None
    item = scan["results"][name]
    if "duration_seconds" not in item:
        return "The checker could not decode this recording.", None
    if around_silence and item["longest_silence_seconds"] > 0:
        offset = max(0, item["longest_silence_start"] - 1)
    offset = max(0, min(float(offset), item["duration_seconds"] - 0.1))
    try:
        result = subprocess.run(
            [_binary("ffmpeg", root), "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(path),
             "-t", "12", "-vn", "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True,
            **_hidden_process_options(),
        )
        samples = np.frombuffer(result.stdout, dtype="<i2").copy()
        if not len(samples):
            return "No audio decoded at this position.", None
        return f"Playing {offset:.2f}s to {offset + len(samples) / 24000:.2f}s.", (24000, samples)
    except (OSError, ValueError, subprocess.SubprocessError):
        return "Could not decode this preview.", None
