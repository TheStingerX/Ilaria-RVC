"""Optional Kokoro ONNX engine, isolated from Ilaria RVC's Python dependencies."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.request import Request, urlopen

KOKORO = "Kokoro (local)"
VOICE_GROUPS = (
    ("af", "en-us", "US English", "alloy aoede bella heart jessica kore nicole nova river sarah sky"),
    ("am", "en-us", "US English", "adam echo eric fenrir liam michael onyx puck santa"),
    ("bf", "en-gb", "UK English", "alice emma isabella lily"),
    ("bm", "en-gb", "UK English", "daniel fable george lewis"),
    ("ef", "es", "Spanish", "dora"),
    ("em", "es", "Spanish", "alex santa"),
    ("ff", "fr-fr", "French", "siwis"),
    ("hf", "hi", "Hindi", "alpha beta"),
    ("hm", "hi", "Hindi", "omega psi"),
    ("if", "it", "Italian", "sara"),
    ("im", "it", "Italian", "nicola"),
    ("jf", "ja", "Japanese", "alpha gongitsune nezumi tebukuro"),
    ("jm", "ja", "Japanese", "kumo"),
    ("pf", "pt-br", "Brazilian Portuguese", "dora"),
    ("pm", "pt-br", "Brazilian Portuguese", "alex santa"),
    ("zf", "cmn", "Mandarin", "xiaobei xiaoni xiaoxiao xiaoyi"),
    ("zm", "cmn", "Mandarin", "yunjian yunxi yunxia yunyang"),
)
VOICES = {
    f"{name.title()} ({label})": (
        f"{prefix}_{name}", locale, "Female" if prefix.endswith("f") else "Male",
    )
    for prefix, locale, label, names in VOICE_GROUPS
    for name in names.split()
}
SOURCE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
FILES = {
    "kokoro-v1.0.onnx": (325505369, "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"),
    "voices-v1.0.bin": (28214398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}
LOCK = threading.Lock()
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def engine_dir(root):
    return Path(root) / "assets" / "tts" / "kokoro"


def installed(root):
    return (engine_dir(root) / "ready-0.6.1").is_file()


def install(root):
    target = engine_dir(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        if installed(root):
            yield "Kokoro is installed and ready."
            return
        if target.exists():
            raise ValueError("An incomplete Kokoro folder already exists. Move it aside before retrying.")
        if shutil.disk_usage(target.parent).free < 1024**3:
            raise ValueError("Free at least 1 GB before installing Kokoro.")
        with tempfile.TemporaryDirectory(prefix=".kokoro-", dir=target.parent) as temporary:
            stage = Path(temporary) / "engine"
            stage.mkdir()
            for name, (size, expected) in FILES.items():
                yield f"Downloading {name} ({size / 1e6:.0f} MB)…"
                digest = hashlib.sha256()
                received = 0
                request = Request(SOURCE + name, headers={"User-Agent": "Ilaria-RVC"})
                with urlopen(request, timeout=60) as response, (stage / name).open("xb") as stream:
                    for block in iter(lambda: response.read(4 << 20), b""):
                        received += len(block)
                        if received > size:
                            raise ValueError(f"Unexpected download size for {name}.")
                        digest.update(block)
                        stream.write(block)
                if received != size or digest.hexdigest() != expected:
                    raise ValueError(f"Download verification failed for {name}.")
            yield "Installing the Kokoro CPU engine…"
            result = subprocess.run(
                [sys.executable, "-I", "-m", "pip", "--isolated", "install",
                 "--disable-pip-version-check", "--no-input", "--only-binary=:all:",
                 "--target", str(stage / "dependencies"), "kokoro-onnx==0.6.1"],
                capture_output=True, text=True, errors="replace", timeout=600, creationflags=NO_WINDOW,
            )
            if result.returncode:
                raise RuntimeError("Kokoro dependency installation failed. " + result.stderr.strip()[-600:])
            _worker(stage, {"check": True}, timeout=60)
            (stage / "ready-0.6.1").touch()
            stage.rename(target)
        yield "Kokoro is installed. Select a source voice and click Generate speech."


def _worker(directory, request, timeout):
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()), str(directory)],
        input=json.dumps(request), capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=NO_WINDOW,
        env={**os.environ, "ONNX_PROVIDER": "CPUExecutionProvider"},
    )
    if result.returncode:
        raise RuntimeError("Kokoro could not generate speech. " + result.stderr.strip()[-600:])


def synthesize(root, text, voice, rate, volume, destination):
    if not installed(root):
        raise ValueError("Click Download and install Kokoro before generating speech.")
    if voice not in VOICES:
        raise ValueError("Choose a Kokoro source voice.")
    _worker(engine_dir(root), {
        "text": text, "voice": voice, "speed": 1 + int(rate) / 100,
        "volume": 1 + int(volume) / 100, "destination": str(destination),
    }, timeout=300)


def _main(directory):
    from wave import open as open_wave

    sys.path.insert(0, str(directory / "dependencies"))
    import numpy as np
    from kokoro_onnx import Kokoro

    request = json.load(sys.stdin)
    engine = Kokoro(str(directory / "kokoro-v1.0.onnx"), str(directory / "voices-v1.0.bin"))
    if request.get("check"):
        return
    voice, language, _ = VOICES[request["voice"]]
    audio, rate = engine.create(request["text"], voice=voice, speed=request["speed"], lang=language)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Kokoro returned empty or invalid audio.")
    pcm = (np.clip(audio * request["volume"], -1, 1) * 32767).astype("<i2")
    with open_wave(request["destination"], "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
