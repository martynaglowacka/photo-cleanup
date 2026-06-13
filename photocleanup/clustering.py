from .distances import hamming


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def cluster(items, threshold):
    names = [name for name, _ in items]
    uf = UnionFind(names)

    for i in range(len(items)):
        name_i, hash_i = items[i]
        for j in range(i + 1, len(items)):
            name_j, hash_j = items[j]
            if hamming(hash_i, hash_j) <= threshold:  # near-duplicate -> link
                uf.union(name_i, name_j)

    groups = {}
    for name in names:
        groups.setdefault(uf.find(name), []).append(name)

    clusters = list(groups.values())
    clusters.sort(key=lambda c: (-len(c), c[0]))  # biggest first
    return clusters
