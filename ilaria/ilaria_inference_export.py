"""Write converted audio in the selected format."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import soundfile


EXPORT_FORMATS = ("wav", "mp3", "flac", "ogg", "m4a")
_LOSSLESS = {"wav", "flac"}
_FFMPEG_CODECS = {
    "mp3": ("mp3", "libmp3lame", "-q:a", "2"),
    "ogg": ("ogg", "libvorbis", "-q:a", "5"),
    "m4a": ("ipod", "aac", "-b:a", "192k"),
}


def write_inference_export(path, samples, rate, ffmpeg=None):
    """Encode a PCM or floating-point array and publish the completed file."""
    destination = Path(path).resolve()
    audio_format = destination.suffix.lower().lstrip(".")
    if audio_format not in EXPORT_FORMATS:
        raise ValueError("Choose WAV, MP3, FLAC, OGG or M4A as the export format.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ilaria-export-", dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        if audio_format in _LOSSLESS:
            soundfile.write(str(staged), samples, int(rate), subtype="PCM_16", format=audio_format.upper())
        else:
            source = Path(directory) / "source.wav"
            soundfile.write(str(source), samples, int(rate), subtype="PCM_16", format="WAV")
            if ffmpeg is None:
                bundled = Path(__file__).resolve().parents[1] / "ffmpeg.exe"
                ffmpeg = bundled if bundled.is_file() else shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("FFmpeg is required to export MP3, OGG and M4A.")
            container, codec, quality_option, quality = _FFMPEG_CODECS[audio_format]
            try:
                result = subprocess.run(
                    [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
                     "-i", str(source), "-vn", "-c:a", codec, quality_option, quality,
                     "-f", container, str(staged)],
                    capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise RuntimeError(f"Could not start FFmpeg to export {audio_format.upper()}: {exc}") from exc
            if result.returncode or not staged.is_file() or not staged.stat().st_size:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                detail = stderr[-1000:] if stderr else "The encoder produced no audio."
                raise RuntimeError(f"Could not export {audio_format.upper()}: {detail}")
        os.replace(staged, destination)
    return destination
