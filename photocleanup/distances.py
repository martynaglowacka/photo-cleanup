def hamming(hash_a, hash_b):
    return hash_a - hash_b  # imagehash overloads '-' to return Hamming distance


def nearest_neighbors(items):
    report = []
    for i, (name_i, hash_i) in enumerate(items):
        best_dist, best_name = None, None
        for j, (name_j, hash_j) in enumerate(items):
            if i == j:
                continue
            d = hamming(hash_i, hash_j)
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, name_j
        report.append((best_dist, name_i, best_name))
    report.sort()
    return report
