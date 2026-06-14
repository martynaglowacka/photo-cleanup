import sys

from .clustering import cluster
from .hashing import hash_folder
from .scoring import score_cluster

DEFAULT_THRESHOLD = 15
DEFAULT_KEEP = 2


def main():
    if len(sys.argv) < 2:
        print("usage: python -m photocleanup <folder> [threshold] [keep]")
        sys.exit(1)

    folder = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    keep = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_KEEP

    items = hash_folder(folder)
    by_name = {p.name: p for p, _ in items}
    clusters = cluster([(p.name, h) for p, h in items], threshold)

    print(f"{len(items)} photos -> {len(clusters)} clusters\n")
    for n, members in enumerate(clusters, 1):
        if len(members) == 1:
            print(f"[single {n}]  keep: {members[0]}\n")
            continue
        ranked = score_cluster([by_name[m] for m in members])
        print(f"[cluster {n}]  {len(members)} photos — keep top {keep}:")
        for i, (score, p) in enumerate(ranked):
            tag = "KEEP  " if i < keep else "delete"
            print(f"    {tag}  {score:.3f}  {p.name}")
        print()


if __name__ == "__main__":
    main()
