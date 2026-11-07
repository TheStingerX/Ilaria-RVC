"""Offline text to speech using the desktop voices installed in Windows."""

from array import array
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave


_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
try {
    $request = ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($env:ILARIA_TTS_REQUEST))
    Add-Type -AssemblyName System.Speech
    $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        if ($request.operation -eq 'list') {
            $voices = @(
                foreach ($voice in $speaker.GetInstalledVoices()) {
                    if ($voice.Enabled) {
                        $info = $voice.VoiceInfo
                        [PSCustomObject]@{
                            ShortName = $info.Name
                            FriendlyName = $info.Name
                            Locale = $info.Culture.Name
                            Gender = $info.Gender.ToString()
                        }
                    }
                }
            )
            ConvertTo-Json -InputObject $voices -Compress
        } else {
            $speaker.SelectVoice([string]$request.voice)
            $speaker.Rate = [int]$request.rate
            $speaker.Volume = 100
            $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
                22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
                [System.Speech.AudioFormat.AudioChannel]::Mono
            )
            $speaker.SetOutputToWaveFile([string]$request.output, $format)
            $speaker.Speak([string]$request.text)
        }
    } finally {
        $speaker.Dispose()
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""
_ENCODED_SCRIPT = base64.b64encode(_SCRIPT.encode("utf-16-le")).decode("ascii")


def _run(request, directory, timeout):
    if os.name != "nt":
        raise RuntimeError("Offline Windows TTS requires Windows.")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / (
        "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    request_path = Path(directory) / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, ILARIA_TTS_REQUEST=str(request_path))
    try:
        result = subprocess.run(
            [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive",
             "-EncodedCommand", _ENCODED_SCRIPT],
            env=env, capture_output=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Offline Windows TTS timed out. Try a shorter passage.") from exc
    except OSError as exc:
        raise RuntimeError("Windows PowerShell is unavailable for offline TTS.") from exc
    if result.returncode:
        detail = result.stderr.split("#< CLIXML", 1)[0].strip()
        if detail:
            raise RuntimeError(f"Offline Windows TTS could not start or synthesize speech. {detail[:300]}")
        raise RuntimeError("Offline Windows TTS failed. Check your installed Windows speech voices.")
    return result.stdout


def list_voices():
    """Return the voices available to System.Speech in this Windows session."""
    with tempfile.TemporaryDirectory(prefix="ilaria-tts-") as directory:
        result = _run({"operation": "list"}, directory, timeout=20)
    try:
        return json.loads(result)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Windows did not return a readable list of speech voices.") from exc


def synthesize(text, voice, rate, volume, output):
    """Save mono PCM WAV. Windows speech rate is an approximate slider."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter text to synthesize.")
    if not voice:
        raise ValueError("Select an installed Windows voice.")
    speed = max(-10, min(10, round(int(rate) / 10)))
    gain = max(0, 1 + int(volume) / 100)
    with tempfile.TemporaryDirectory(prefix="ilaria-tts-") as directory:
        temporary = Path(directory) / "speech.wav"
        _run({"operation": "synthesize", "text": text, "voice": voice,
              "rate": speed, "output": str(temporary)}, directory, timeout=180)
        with wave.open(str(temporary), "rb") as source:
            if not source.getnframes():
                raise RuntimeError("Offline Windows TTS produced no audio.")
            if source.getsampwidth() != 2:
                raise RuntimeError("Offline Windows TTS returned an unsupported audio format.")
            if gain != 1:
                with wave.open(str(output), "wb") as target:
                    target.setparams(source.getparams())
                    while True:
                        samples = array("h", source.readframes(8192))
                        if not samples:
                            break
                        scaled = array("h", (max(-32768, min(32767, round(sample * gain)))
                                             for sample in samples))
                        target.writeframesraw(scaled.tobytes())
        if gain == 1:
            shutil.copyfile(temporary, output)
