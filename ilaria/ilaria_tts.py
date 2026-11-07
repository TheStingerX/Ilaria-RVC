"""Speech synthesis for the local Inference tab."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

EDGE = "Microsoft Edge (online)"
WINDOWS = "Windows voices (offline)"
MAX_TEXT = 10000
ALL_LANGUAGES = "All languages"
ALL_GENDERS = "Any voice"
DEFAULT_VOICES = json.loads(Path(__file__).with_name("tts_voices.json").read_text(encoding="utf-8-sig"))
LANGUAGES = {item["Locale"].split("-")[0]: item["Language"] for item in DEFAULT_VOICES}


def language(voice):
    locale = voice["Locale"]
    return LANGUAGES.get(locale.split("-")[0], locale)


def _run(command, timeout, request=None):
    return subprocess.run(
        command, input=request, capture_output=True, encoding="utf-8",
        errors="replace", timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _edge(request, timeout):
    result = _run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--edge-worker"],
        timeout, json.dumps(request),
    )
    if result.returncode:
        detail = result.stdout.strip()[:320]
        raise RuntimeError("Edge TTS could not connect or complete the request. "
                           "Check your connection or try Windows voices. " + detail)
    return json.loads(result.stdout)


def voices(engine, refresh=False):
    if engine == WINDOWS:
        from ilaria_tts_windows import list_voices

        return list_voices()
    if engine != EDGE:
        raise ValueError("Choose a speech engine.")
    if refresh:
        return _edge({"operation": "list"}, 25)
    return DEFAULT_VOICES


def voice_choices(catalog, locale, gender):
    return sorted({
        voice["ShortName"] for voice in catalog
        if (locale == ALL_LANGUAGES or language(voice) == locale)
        and (gender == ALL_GENDERS or voice["Gender"] == gender)
    }, key=str.casefold)


def read_text(upload, current):
    if upload is None:
        return current, "Choose a UTF-8 .txt file to import."
    try:
        with open(upload.name, "rb") as source:
            data = source.read(MAX_TEXT * 4 + 4)
        if len(data) >= MAX_TEXT * 4 + 4:
            raise ValueError("Text exceeds the 10,000 character limit.")
        value = data.decode("utf-8-sig")
        if len(value) > MAX_TEXT:
            raise ValueError("Text exceeds the 10,000 character limit.")
        return value, f"Loaded {len(value):,} characters. Edit the text before generating if needed."
    except (OSError, UnicodeError):
        return current, "Could not read the file. Save it as a UTF-8 .txt file."
    except ValueError as exc:
        return current, str(exc)


def _encode(root, source, target, codec):
    result = _run(
        [str(root / "ffmpeg.exe"), "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(source), "-c:a", codec, str(target)],
        45,
    )
    if result.returncode or not target.is_file() or not target.stat().st_size:
        raise RuntimeError("Could not encode audio. Check that ffmpeg.exe is available.")


def generate(root, convert, text, engine, voice, rate, volume, source_pitch,
             apply_rvc, output_format):
    """Yield status, source player, RVC player and download."""
    text = (text or "").strip()
    if not text or len(text) > MAX_TEXT:
        yield "Enter between 1 and 10,000 characters.", None, None, None
        return
    if not voice:
        yield "Select a source voice. Refresh voices to see available choices.", None, None, None
        return
    codecs = {"WAV": "pcm_s16le", "FLAC": "flac", "MP3": "libmp3lame"}
    if output_format not in codecs:
        yield "Choose WAV, FLAC or MP3.", None, None, None
        return
    start = time.monotonic()
    source = converted = None
    yield f"Generating speech with {voice}…", None, None, None
    try:
        root = Path(root)
        result_dir = Path(tempfile.mkdtemp(prefix="tts-output-", dir=root / "TEMP"))
        with tempfile.TemporaryDirectory(prefix="tts-work-", dir=root / "TEMP") as scratch:
            source_path = result_dir / "speech.wav"
            if engine == EDGE:
                raw = Path(scratch) / "source.mp3"
                _edge({"operation": "synthesize", "text": text, "voice": voice,
                       "rate": f"{int(rate):+d}%", "volume": f"{int(volume):+d}%",
                       "pitch": f"{int(source_pitch):+d}Hz", "output": str(raw)}, 185)
                _encode(root, raw, source_path, "pcm_s16le")
            elif engine == WINDOWS:
                from ilaria_tts_windows import synthesize

                synthesize(text, voice, rate, volume, source_path)
            else:
                raise ValueError("Choose a speech engine.")
            source = str(source_path)
            final_path = source_path
            model_message = ""
            if apply_rvc:
                yield "Speech ready. Applying the selected RVC model…", source, None, None
                model_message, audio = convert(source)
                if audio is None:
                    yield model_message, source, None, None
                    return
                import soundfile

                final_path = result_dir / "converted.wav"
                sample_rate, samples = audio
                soundfile.write(str(final_path), samples, sample_rate, subtype="PCM_16")
                converted = str(final_path)
            download = final_path
            if output_format != "WAV":
                download = final_path.with_suffix("." + output_format.lower())
                _encode(root, final_path, download, codecs[output_format])
        status = f"Done in {time.monotonic() - start:.1f} s."
        yield status + ("\n" + model_message if model_message else ""), source, converted, str(download)
    except subprocess.TimeoutExpired:
        yield "Speech generation timed out. Try a shorter passage or the offline engine.", source, converted, None
    except (RuntimeError, ValueError, OSError) as exc:
        yield str(exc), source, converted, None


async def _edge_worker(request):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tts_vendor"))
    import edge_tts

    if request["operation"] == "list":
        result = await asyncio.wait_for(edge_tts.list_voices(), 20)
        return [{key: voice[key] for key in ("ShortName", "Locale", "Gender")} for voice in result]
    speech = edge_tts.Communicate(
        request["text"], request["voice"], rate=request["rate"],
        volume=request["volume"], pitch=request["pitch"],
    )
    await asyncio.wait_for(speech.save(request["output"]), 175)
    await asyncio.sleep(0.25)
    return True


def build_tab(gr, root, model, convert, track, pitch_methods, index_paths):
    """Build the TTS subtab and return its shared model controls."""
    def catalog_controls(catalog, locale, gender, voice):
        available = voice_choices(catalog, locale, gender)
        selected = voice if voice in available else next(iter(available), None)
        return gr.Dropdown.update(choices=available, value=selected)

    def load_catalog(engine, current, locale, gender, voice, refresh):
        try:
            catalog = voices(engine, refresh)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            catalog = current if engine == EDGE else []
            message = f"Could not refresh voices. {error}"
        else:
            message = f"{len(catalog)} voices available." if catalog else "Install a Windows desktop speech voice to use this engine."
        locales = [ALL_LANGUAGES] + sorted({language(item) for item in catalog})
        genders = [ALL_GENDERS] + sorted({item["Gender"] for item in catalog})
        locale = locale if locale in locales else ALL_LANGUAGES
        gender = gender if gender in genders else ALL_GENDERS
        return (catalog, gr.Dropdown.update(choices=locales, value=locale),
                gr.Dropdown.update(choices=genders, value=gender),
                catalog_controls(catalog, locale, gender, voice),
                gr.Slider.update(visible=engine == EDGE), message)

    def generate_audio(text, engine, voice, rate, volume, source_pitch, apply_rvc,
                       selected_model, speaker, semitones, method, index,
                       ratio, median, sample_rate, envelope, protect):
        if apply_rvc and not selected_model:
            yield ("Select a Voice Model above or turn off RVC conversion.",
                   gr.update(value=None, visible=True), gr.update(value=None, visible=True))
            return

        def apply_model(path):
            return convert(selected_model, speaker, path, semitones, method, index,
                           ratio, median, sample_rate, envelope, protect)

        for status, source, converted, _ in generate(
            root, apply_model, text, engine, voice, rate, volume,
            source_pitch, apply_rvc, "WAV",
        ):
            yield (
                status,
                gr.update(value=source, visible=True),
                gr.update(value=converted, visible=True),
            )

    with gr.TabItem("TTS", elem_id="tts-tab"):
        gr.Markdown(
            "### Text to speech\n"
            "Write or import text, choose a source voice, then generate speech. "
            "The full Edge catalog is available immediately. Use **Language** to filter it. Enable RVC conversion to use the **Voice Model** selected above.\n\n"
            "**Microsoft Edge** sends the text to Microsoft's online service. "
            "**Windows voices** use installed desktop voices and work offline.",
            elem_id="tts-intro",
        )
        text = gr.Textbox(label="Text to speak", lines=5, max_lines=12,
                          placeholder="Type here (up to 10,000 characters)…")
        with gr.Accordion("Import text from a file", open=False, elem_id="tts-import-text"):
            upload = gr.File(label="UTF-8 text file", file_types=[".txt"], type="file")
            import_button = gr.Button("Load text")
        catalog = gr.State(DEFAULT_VOICES)
        with gr.Row():
            engine = gr.Radio([EDGE, WINDOWS], value=EDGE, label="Speech engine")
            locale = gr.Dropdown([ALL_LANGUAGES] + sorted({language(item) for item in DEFAULT_VOICES}), value=ALL_LANGUAGES,
                                 label="Language")
            gender = gr.Dropdown([ALL_GENDERS, "Female", "Male"], value=ALL_GENDERS,
                                 label="Voice filter")
        with gr.Row():
            voice = gr.Dropdown(voice_choices(DEFAULT_VOICES, ALL_LANGUAGES, ALL_GENDERS),
                                value="it-IT-ElsaNeural", label="Source voice")
            refresh = gr.Button("Refresh voices", variant="primary")
        with gr.Accordion("Speech settings", open=False, elem_id="tts-speech-settings"):
            with gr.Row():
                rate = gr.Slider(-50, 100, value=0, step=5, label="Speaking rate adjustment (%)")
                volume = gr.Slider(-80, 50, value=0, step=5, label="Source volume adjustment (%)")
                source_pitch = gr.Slider(-100, 100, value=0, step=5, label="Source pitch adjustment (Hz)")
        with gr.Row():
            apply_rvc = gr.Checkbox(label="Convert with selected RVC model", value=True)
        with gr.Accordion("RVC settings", open=False, elem_id="tts-rvc-settings"):
            with gr.Row():
                speaker = gr.Slider(0, 2333, value=0, step=1, label="Speaker ID", visible=False)
                pitch = gr.Slider(-24, 24, value=0, step=1, label="RVC pitch shift (semitones)")
                method = gr.Dropdown(pitch_methods, value="rmvpe", label="Pitch extraction")
            with gr.Row():
                index = gr.Dropdown(sorted(index_paths), label="Voice index")
                ratio = gr.Slider(0, 1, value=0, step=0.01, label="Index ratio")
            with gr.Row():
                median = gr.Slider(0, 7, value=3, step=1, label="Harvest median filter")
                sample_rate = gr.Dropdown(["Model default", "16000", "22050", "24000", "32000",
                                           "40000", "44100", "48000"], value="Model default",
                                          label="RVC output sample rate (Hz)")
                envelope = gr.Slider(0, 1, value=0.25, step=0.01, label="Volume envelope mix")
                protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Consonant protection")
        generate_button = gr.Button("Generate speech", variant="primary")
        with gr.Row():
            source_audio = gr.Audio(label="TTS Result", interactive=False, elem_id="tts-source-audio")
            rvc_audio = gr.Audio(label="RVC Result", interactive=False)
        log = gr.Textbox(label="TTS+RVC Log", lines=3, max_lines=9, interactive=False)

        import_button.click(read_text, [upload, text], [text, log], show_progress=False)
        catalog_inputs = [engine, catalog, locale, gender, voice]
        catalog_outputs = [catalog, locale, gender, voice, source_pitch, log]
        engine.change(lambda *args: load_catalog(*args, False), catalog_inputs, catalog_outputs)
        refresh.click(lambda *args: load_catalog(*args, True), catalog_inputs, catalog_outputs)
        for filter_control in (locale, gender):
            filter_control.change(catalog_controls, [catalog, locale, gender, voice], voice,
                                  show_progress=False)
        generate_button.click(
            track(generate_audio, "text to speech"),
            [text, engine, voice, rate, volume, source_pitch, apply_rvc,
             model, speaker, pitch, method, index, ratio, median, sample_rate, envelope, protect],
            [log, source_audio, rvc_audio],
        )
    return speaker, index


if __name__ == "__main__" and sys.argv[1:] == ["--edge-worker"]:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        print(json.dumps(asyncio.run(_edge_worker(json.load(sys.stdin)))))
    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
