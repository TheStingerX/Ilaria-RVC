"""Feature reduction choices for the FAISS retrieval index, independent of PTH training."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import math

ALGORITHMS = ('Auto', 'Faiss', 'KMeans')
AUTO_CLUSTER_THRESHOLD = 200000
MAX_CENTERS = 10000


def cluster_count(rows, algorithm):
    """Zero means retain every feature row; explicit KMeans reduces smaller sets too."""
    if algorithm not in ALGORITHMS:
        raise ValueError('Choose Index Algorithm: Auto, Faiss or KMeans.')
    if rows <= 0:
        raise ValueError('No audio features found. Run Feature Extraction first.')
    if algorithm == 'Faiss' or (algorithm == 'Auto' and rows <= AUTO_CLUSTER_THRESHOLD):
        return 0
    if rows <= 39:
        return 0  # Preserve the already tiny retrieval set instead of collapsing it.
    if algorithm == 'Auto':
        return MAX_CENTERS
    return min(MAX_CENTERS, max(39, rows // 10), rows - 1)


def ivf_list_count(rows):
    if rows <= 0:
        raise ValueError('Cannot create an index without features.')
    return max(1, min(int(16 * math.sqrt(rows)), rows // 39))


def prepare_index_features(features, algorithm='Auto', cpu_threads=8):
    """Yield one-line progress; return the retained or clustered float32 matrix.

    Auto preserves the existing >200k/10k-centers heuristic, including fallback
    on clustering failure. Explicit KMeans reports failure instead of pretending
    it ran. All algorithms still create a normal FAISS index in the caller.
    """
    import numpy as np
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError('Audio features must be a matrix with a nonzero feature dimension.')
    rows = len(features)
    centers = cluster_count(rows, algorithm)
    if not centers:
        reason = 'too few features for useful clustering' if algorithm == 'KMeans' else 'keeping all audio features'
        yield 'Train Index: %s - %s (%s vectors) | 20.0%%' % (algorithm, reason, rows)
        return features
    yield 'Train Index: %s - clustering %s feature vectors into %s centers... | 20.0%%' % (algorithm, rows, centers)
    try:
        from sklearn.cluster import MiniBatchKMeans
        reduced = MiniBatchKMeans(n_clusters=centers, verbose=False,
                                  batch_size=256 * max(1, int(cpu_threads)),
                                  compute_labels=False, init='random', n_init=3).fit(features).cluster_centers_
        return np.ascontiguousarray(reduced, dtype=np.float32)
    except Exception as error:
        if algorithm != 'Auto':
            raise ValueError('KMeans clustering failed: ' + str(error)) from error
        yield 'Train Index: Auto clustering failed; keeping original features... | 20.0%'
        return features
