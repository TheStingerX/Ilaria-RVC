"""Optional pitch and silence processing for RVC inference."""

from bisect import bisect_right

import numpy as np


# Equal-tempered notes from G1 through C6.
CHROMATIC_NOTES = 440.0 * np.exp2(np.arange(-38, 16) / 12)


def apply_pitch_options(
    f0, semitones, autotune_strength=None, proposed_pitch_hz=None, analysis_slice=None
):
    """Adjust voiced pitch before RVC quantizes it for the synthesizer."""
    f0 = np.asarray(f0)
    adjusted = f0.astype(np.result_type(f0.dtype, np.float32), copy=True)
    voiced = np.isfinite(adjusted) & (adjusted > 0)
    adjusted[~voiced] = 0
    if autotune_strength:
        frequencies = adjusted[voiced]
        upper = np.searchsorted(CHROMATIC_NOTES, frequencies).clip(1, len(CHROMATIC_NOTES) - 1)
        lower_note = CHROMATIC_NOTES[upper - 1]
        upper_note = CHROMATIC_NOTES[upper]
        closest = np.where(
            frequencies - lower_note <= upper_note - frequencies, lower_note, upper_note
        )
        adjusted[voiced] += (closest - frequencies) * autotune_strength

    if proposed_pitch_hz is not None:
        analysis = adjusted if analysis_slice is None else adjusted[analysis_slice]
        voiced_frames = np.flatnonzero(analysis > 0)
        if len(voiced_frames) >= 2:
            contour = np.interp(np.arange(len(analysis)), voiced_frames, analysis[voiced_frames])
            median = np.median(contour)
            offset = int(np.clip(np.rint(12 * np.log2(proposed_pitch_hz / median)), -12, 12))
            semitones += offset
    return adjusted * np.exp2(semitones / 12)


def silence_split_points(audio, sample_rate, frame_size, existing_points=()):
    """Add frame-aligned cuts inside pauses, retaining the normal long-audio cuts."""
    length = len(audio)
    aligned = {int(point) // frame_size * frame_size for point in existing_points}
    existing = sorted(point for point in aligned if 0 < point < length)
    min_chunk = sample_rate
    if length < 2 * min_chunk or not np.any(audio):
        return existing

    from librosa.effects import split

    silence_frame = max(2, round(sample_rate * 0.25))
    voiced_intervals = split(
        audio, top_db=60, frame_length=silence_frame, hop_length=silence_frame // 2
    )
    bounds = [0, *existing, length]
    added = []
    last_added = 0
    for left, right in zip(voiced_intervals[:-1, 1], voiced_intervals[1:, 0]):
        point = int((left + right) // (2 * frame_size) * frame_size)
        next_cut = bisect_right(bounds, point)
        if point <= left or point >= right or next_cut == len(bounds):
            continue
        previous_cut = max(bounds[next_cut - 1], last_added)
        if point - previous_cut >= min_chunk and bounds[next_cut] - point >= min_chunk:
            added.append(point)
            last_added = point
    return sorted(existing + added)
