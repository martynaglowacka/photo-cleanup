import sys

from .clustering import cluster
from .hashing import hash_folder

DEFAULT_THRESHOLD = 15


def main():
    if len(sys.argv) < 2:
        print("usage: python -m photocleanup <folder> [threshold]")
        sys.exit(1)

    folder = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    items = [(p.name, h) for p, h in hash_folder(folder)]
    clusters = cluster(items, threshold)

    print(f"{len(items)} photos -> {len(clusters)} clusters "
          f"(threshold = {threshold} bits)\n")
    for n, members in enumerate(clusters, 1):
        label = "cluster" if len(members) > 1 else "single"
        print(f"[{label} {n}]  {len(members)} photo(s)")
        for name in members:
            print(f"    {name}")
        print()


if __name__ == "__main__":
    main()
