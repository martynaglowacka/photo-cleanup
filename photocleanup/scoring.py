from .ranking import exposure, eyes_open, sharpness

WEIGHTS = {"eyes": 0.5, "sharpness": 0.35, "exposure": 0.15}
REASONS = {"eyes": "eyes open", "sharpness": "sharpest", "exposure": "best exposed"}


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
        "eyes": _normalize([eyes_open(p) for p in paths]),
        "sharpness": _normalize([sharpness(p) for p in paths]),
        "exposure": _normalize([exposure(p) for p in paths]),
    }
    rows = []
    for i, p in enumerate(paths):
        signals = {k: norm[k][i] for k in WEIGHTS}
        total = wsum = 0.0
        for k, w in WEIGHTS.items():
            if signals[k] is not None:  # missing signal: drop weight, renormalize
                total += w * signals[k]
                wsum += w
        rows.append({"path": p, "score": total / wsum if wsum else 0.0, "signals": signals})
    rows.sort(key=lambda r: -r["score"])
    return rows


def reasons(signals):
    present = {k: v for k, v in signals.items() if v is not None}
    strong = [k for k in sorted(present, key=lambda k: -present[k]) if present[k] >= 0.6]
    return [REASONS[k] for k in strong[:2]]


def allocate(sizes, total_keep):
    if not sizes:
        return []
    grand = sum(sizes)
    keeps = [max(1, round(total_keep * s / grand)) for s in sizes]  # proportional, >=1 each
    return [min(k, s) for k, s in zip(keeps, sizes)]
