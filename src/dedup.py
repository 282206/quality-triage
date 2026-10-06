"""Phase 2 - near-duplicate grouping with perceptual hashes + union-find.

pHash: DCT of a 32x32 grayscale thumbnail, keep the 8x8 lowest frequencies,
threshold at their median -> 64-bit hash. Similar-looking images have hashes
that differ in few bits (small Hamming distance), even after resizing,
re-compression or small exposure changes.

Two images are linked if Hamming distance <= threshold (default 8 of 64 bits).
Union-find turns those pairwise links into groups (transitive: if A~B and
B~C then A, B, C are one group).

Run:  python -m src.dedup [folder] [--threshold 8]
"""
from __future__ import annotations

import argparse

import numpy as np

DEFAULT_THRESHOLD = 8


class UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def hex_to_bits(h: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))


def hamming_matrix(hashes: list[str]) -> np.ndarray:
    """All-pairs Hamming distances (n x n). Fine for a few thousand photos."""
    bits = np.stack([hex_to_bits(h) for h in hashes]).astype(np.uint8)
    # distance = number of differing bits = sum(a xor b)
    return (bits[:, None, :] != bits[None, :, :]).sum(-1)


def group_duplicates(hashes: list[str], threshold: int = DEFAULT_THRESHOLD) -> list[int]:
    """Return a group id per image. Images without a duplicate get group -1."""
    n = len(hashes)
    if n == 0:
        return []
    d = hamming_matrix(hashes)
    uf = UnionFind(n)
    ii, jj = np.where(np.triu(d <= threshold, k=1))
    for i, j in zip(ii, jj):
        uf.union(int(i), int(j))
    roots = [uf.find(i) for i in range(n)]
    counts = {r: roots.count(r) for r in set(roots)}
    ids, out = {}, []
    for r in roots:
        if counts[r] < 2:
            out.append(-1)
        else:
            out.append(ids.setdefault(r, len(ids)))
    return out


def main(argv=None):
    from src.common import RAW_DIR
    from src.features import raw_features

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)
    df = raw_features(RAW_DIR)
    df["dup_group"] = group_duplicates(df.phash.astype(str).tolist(), args.threshold)
    groups = df[df.dup_group >= 0].groupby("dup_group")["filename"].apply(list)
    print(f"{len(groups)} duplicate groups (threshold={args.threshold}):")
    for g, names in groups.items():
        print(f"  group {g}: {names}")


if __name__ == "__main__":
    main()
