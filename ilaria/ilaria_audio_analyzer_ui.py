"""Audio Analyzer panel for the bundled Gradio 3.14 runtime."""

OVERVIEW_HEADERS = [
    "File", "Result", "Format / codec", "Duration (s)", "Analyzed (s)",
    "Sample rate (Hz)", "Channels", "Peak (dBFS)", "RMS (dBFS)",
    "Silence (%)", "Near full scale (%)",
]
CHANNEL_HEADERS = [
    "Channel", "Peak (dBFS)", "RMS (dBFS)", "DC offset (%)",
    "Near full scale (%)",
]
SILENCE_HEADERS = ["Start (s)", "End (s)", "Duration (s)"]


def _table(headers, rows=None):
    # Gradio 3.14 indexes the first row of raw lists, including empty lists.
    return {"headers": headers, "data": [] if rows is None else rows}


def build_audio_analyzer(gr, root, track):
    """Add a collapsed analyzer panel at the caller's current location."""
    def analyze(uploads, start, end, silence_db, min_silence, clip_db, channel,
                scale, n_fft, db_range, show_pitch, pitch_min, pitch_max,
                export_format):
        from ilaria_audio_analyzer import analyze_uploads

        for status, overview, detail, channels, silence, plot, preview, report in analyze_uploads(
            root, uploads, start, end, silence_db, min_silence, clip_db,
            channel, scale, int(n_fft), db_range, show_pitch, pitch_min,
            pitch_max, export_format,
        ):
            yield (
                status, _table(OVERVIEW_HEADERS[:7], [row[:7] for row in overview]),
                _table([OVERVIEW_HEADERS[0], *OVERVIEW_HEADERS[7:]],
                       [[row[0], *row[7:]] for row in overview]), detail,
                _table(CHANNEL_HEADERS, channels), _table(SILENCE_HEADERS, silence),
                plot, preview, report,
            )

    with gr.Accordion("Audio Analyzer", open=False, elem_id="audio-analyzer"):
        gr.Markdown(
            "Compare up to **8 local audio files** for recording levels, silence, "
            "channel balance and frequency content. The originals stay intact. "
            "You can also save a report to compare recordings later.",
            elem_id="audio-analyzer-help",
        )
        uploads = gr.File(
            label="Audio files (up to 8)", file_count="multiple",
            file_types=["audio"],
            type="file", interactive=True,
        )
        gr.Markdown(
            "The analyzer reads up to 180 seconds and 12 million individual samples "
            "per file, counting all channels. It displays the analyzed duration in the table."
        )
        with gr.Accordion("Analysis settings", open=False, elem_id="audio-analyzer-settings"):
            gr.Markdown("**Time range and thresholds**")
            with gr.Row():
                start = gr.Number(label="Start (seconds)", value=0)
                end = gr.Number(label="End (seconds, 0 = automatic)", value=0)
                export_format = gr.Dropdown(
                    label="Report format", choices=["None", "CSV", "JSON", "ZIP"], value="ZIP",
                )
            with gr.Row():
                silence_db = gr.Slider(
                    label="Silence threshold (dBFS)", minimum=-90, maximum=-10,
                    value=-45, step=1,
                )
                min_silence = gr.Slider(
                    label="Minimum silence duration (seconds)", minimum=0.05, maximum=10,
                    value=0.3, step=0.05,
                )
                clip_db = gr.Slider(
                    label="Near-full-scale threshold (dBFS)", minimum=-6, maximum=0,
                    value=-0.1, step=0.1,
                )
            gr.Markdown(
                "Near-full-scale counts samples above the chosen level. "
                "Listen to the recording to confirm distortion."
            )
            gr.Markdown("**Spectrum and pitch**")
            with gr.Row():
                channel = gr.Dropdown(
                    label="Channel for plots", choices=["Mix", "Left", "Right", "Side"], value="Mix",
                )
                scale = gr.Dropdown(
                    label="Frequency scale", choices=["Linear", "Log", "Mel"], value="Log",
                )
                n_fft = gr.Dropdown(
                    label="FFT window (samples)",
                    choices=["512", "1024", "2048", "4096"], value="1024",
                )
                db_range = gr.Slider(
                    label="Spectrogram contrast (dB)", minimum=30, maximum=100,
                    value=70, step=5,
                )
            gr.Markdown(
                "**Mix** combines channels for the plots. **Side** shows differences "
                "between the left and right channels. Longer FFT windows separate nearby "
                "frequencies more clearly; shorter windows show brief changes more clearly."
            )
            with gr.Row():
                show_pitch = gr.Checkbox(label="Estimate pitch", value=True)
                pitch_min = gr.Number(label="Lowest pitch (Hz)", value=70)
                pitch_max = gr.Number(label="Highest pitch (Hz)", value=650)
            gr.Markdown(
                "Pitch estimation works best for one clear voice. Enable it if you want "
                "to inspect the pitch of the detailed preview."
            )
        run = gr.Button("Analyze audio", variant="primary")
        status = gr.Textbox(
            label="Analysis Log", value="Add audio files, then click Analyze audio.",
            lines=2, max_lines=5, interactive=False,
        )
        metadata_headers = OVERVIEW_HEADERS[:7]
        level_headers = [OVERVIEW_HEADERS[0], *OVERVIEW_HEADERS[7:]]
        overview = gr.Dataframe(
            label="File details", headers=metadata_headers, value=_table(metadata_headers),
            datatype=["str"] * 3 + ["number"] * 4, interactive=False, max_rows=8, wrap=True,
        )
        levels = gr.Dataframe(
            label="Levels and silence", headers=level_headers, value=_table(level_headers),
            datatype=["str"] + ["number"] * 4, interactive=False, max_rows=8, wrap=True,
        )
        report = gr.File(label="Download analysis report", interactive=False)
        gr.Markdown(
            "### Detailed preview\n"
            "The plot and detailed measurements show the first file compatible with the selected view. "
            "Use the player to hear up to 30 seconds of the analyzed section."
        )
        detail = gr.Markdown("", elem_id="audio-analyzer-detail")
        preview = gr.Audio(label="Audio preview (up to 30 seconds)", interactive=False)
        plot = gr.Image(label="Audio analysis plots", type="filepath", interactive=False)
        with gr.Row():
            channels = gr.Dataframe(
                label="Channel measurements", headers=CHANNEL_HEADERS, value=_table(CHANNEL_HEADERS),
                datatype=["str"] + ["number"] * 4, interactive=False, max_rows=8, wrap=True,
            )
            silence = gr.Dataframe(
                label="Silent intervals", headers=SILENCE_HEADERS, value=_table(SILENCE_HEADERS),
                datatype=["number"] * 3, interactive=False, max_rows=10,
            )

        run.click(
            track(analyze, "audio analysis"),
            [
                uploads, start, end, silence_db, min_silence, clip_db, channel,
                scale, n_fft, db_range, show_pitch, pitch_min, pitch_max, export_format,
            ],
            [status, overview, levels, detail, channels, silence, plot, preview, report],
            show_progress=False,
        )
