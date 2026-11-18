"""Optional two-pass high-register RMVPE correction, inspired by Applio."""

import numpy as np
from scipy.signal import resample_poly


SENSITIVITY = {"Conservative": 80, "Balanced": 150, "High": 250}
SAMPLE_RATE = 16000
HOP = 160


def correct_high_register(audio, original, estimate, mode, ceiling, threshold, sensitivity, minimum_ms):
    """Use a half-speed estimate to recover sustained notes above the threshold."""
    original = np.asarray(original)
    corrected = original.copy()
    guide = np.zeros(len(original), dtype=np.float32)
    block = 1500
    context = 24
    for start in range(0, len(guide), block):
        end = min(start + block, len(guide))
        left = max(0, start - context)
        samples = audio[left * HOP:(end + context) * HOP]
        if len(samples) < HOP:
            continue
        slower = resample_poly(samples, 2, 1)
        pitch = np.asarray(estimate(slower))[::2] * 2
        segment = pitch[start - left:end - left]
        guide[start:start + len(segment)] = segment

    valid = np.isfinite(guide) & (guide >= threshold)
    if mode == "true_pitch":
        valid &= guide <= ceiling
    voiced = np.isfinite(original) & (original > 0)
    cents = np.full(len(original), np.inf)
    comparable = voiced & valid
    cents[comparable] = np.abs(1200 * np.log2(original[comparable] / guide[comparable]))
    tolerance = SENSITIVITY[sensitivity]
    octave_error = np.abs(cents - 1200) <= tolerance
    inconsistent = cents > tolerance
    # Conservative mode requires an octave match. Higher sensitivity can also recover missing notes.
    candidates = valid & (octave_error | (inconsistent & (sensitivity != "Conservative")))
    if mode == "fold":
        # Frames that already carry the correct folded pitch need no replacement.
        if sensitivity == "Conservative":
            candidates = valid & voiced & (np.abs(cents - 1200 * np.log2(3)) <= tolerance)
        else:
            candidates &= ~octave_error
    candidates &= np.abs(original - (guide if mode == "true_pitch" else guide / 2)) > 0.01
    edges = np.flatnonzero(np.diff(np.r_[False, candidates, False]))
    minimum = max(1, int(np.ceil(minimum_ms / 10)))
    divisor = 1 if mode == "true_pitch" else 2
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= minimum:
            corrected[start:end] = guide[start:end] / divisor
    return corrected
