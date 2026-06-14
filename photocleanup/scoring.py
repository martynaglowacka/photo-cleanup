from .ranking import exposure, eyes_open, sharpness
# arbitrary choice for now
WEIGHTS = {"eyes": 0.5, "sharpness": 0.35, "exposure": 0.15}


def _normalize(values):
    present = [v for v in values if v is not None]
    if not present:
        return list(values)
    lo, hi = min(present), max(present)
    if hi == lo:
        return [None if v is None else 1.0 for v in values]
    return [None if v is None else (v - lo) / (hi - lo) for v in values]


def score_cluster(paths):
    norm = {
        "sharpness": _normalize([sharpness(p) for p in paths]),
        "exposure": _normalize([exposure(p) for p in paths]),
        "eyes": _normalize([eyes_open(p) for p in paths]),
    }
    ranked = []
    for i, p in enumerate(paths):
        total = wsum = 0.0
        for k, w in WEIGHTS.items():
            v = norm[k][i]
            if v is not None:  # missing signal
                total += w * v
                wsum += w
        ranked.append((total / wsum if wsum else 0.0, p))
    ranked.sort(key=lambda x: -x[0])
    return ranked
