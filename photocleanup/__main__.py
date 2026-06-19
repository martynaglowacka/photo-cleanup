import sys

from .clustering import cluster
from .hashing import hash_folder
from .scoring import allocate, reasons, score_cluster

DEFAULT_THRESHOLD = 15
DEFAULT_KEEP = 4  # global target across the whole library


def main():
    if len(sys.argv) < 2:
        print("usage: python -m photocleanup <folder> [threshold] [keep_total]")
        sys.exit(1)

    folder = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    total_keep = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_KEEP

    items = hash_folder(folder)
    by_name = {p.name: p for p, _ in items}
    groups = cluster([(p.name, h) for p, h in items], threshold)
    multi = [g for g in groups if len(g) > 1]
    singles = [g for g in groups if len(g) == 1]
    keeps_per = allocate([len(g) for g in multi], total_keep)

    print(f"{len(items)} photos -> {len(groups)} clusters; target keep ~{total_keep}\n")
    for n, (members, k) in enumerate(zip(multi, keeps_per), 1):
        rows = score_cluster([by_name[m] for m in members])
        print(f"[cluster {n}]  {len(members)} photos, keep {k}:")
        for i, r in enumerate(rows):
            tag = "KEEP  " if i < k else "delete"
            why = "  (" + ", ".join(reasons(r["signals"])) + ")" if i < k else ""
            print(f"    {tag} {r['score']:.3f}  {r['path'].name}{why}")
        print()
    if singles:
        print(f"[maybe]  {len(singles)} unique photos — your call:")
        for g in singles:
            print(f"    maybe  {g[0]}")


if __name__ == "__main__":
    main()
