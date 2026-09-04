"""All 4 PS-required interleaver types: block, convolutional, diagonal,
pseudo-random. Each exposes interleave()/deinterleave() as exact inverses."""
import numpy as np


def block_interleave(bits: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    n = rows * cols
    pad = (-len(bits)) % n
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    grid = padded.reshape(-1, rows, cols)
    out = grid.transpose(0, 2, 1).reshape(-1)
    return out


def block_deinterleave(bits: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    n = rows * cols
    pad = (-len(bits)) % n
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    grid = padded.reshape(-1, cols, rows)
    out = grid.transpose(0, 2, 1).reshape(-1)
    return out[: len(bits)]


def convolutional_interleave(bits: np.ndarray, n_branches: int = 8, delay_step: int = 4) -> np.ndarray:
    """Shift-register diagonal-delay interleaver (Ramsey type II)."""
    branches = [[] for _ in range(n_branches)]
    out = []
    delays = [i * delay_step for i in range(n_branches)]
    max_delay = max(delays)
    queues = [list(np.zeros(d, dtype=bits.dtype)) for d in delays]
    idx = 0
    total = len(bits) + n_branches * max_delay
    bit_iter = iter(bits)
    branch = 0
    produced = 0
    padded_bits = list(bits) + [0] * (n_branches * max_delay)
    for b in padded_bits:
        q = queues[branch]
        q.append(b)
        out.append(q.pop(0))
        branch = (branch + 1) % n_branches
    return np.array(out, dtype=bits.dtype)


def convolutional_deinterleave(bits: np.ndarray, n_branches: int = 8, delay_step: int = 4) -> np.ndarray:
    delays = [(n_branches - 1 - i) * delay_step for i in range(n_branches)]
    max_delay = max(delays)
    queues = [list(np.zeros(d, dtype=bits.dtype)) for d in delays]
    out = []
    branch = 0
    for b in bits:
        q = queues[branch]
        q.append(b)
        out.append(q.pop(0))
        branch = (branch + 1) % n_branches
    trimmed = out[n_branches * max_delay:]
    return np.array(trimmed, dtype=bits.dtype)


def _diagonal_perm(n: int, rows: int, cols: int) -> np.ndarray:
    idx = np.arange(rows * cols)
    grid = idx.reshape(rows, cols)
    perm = np.zeros(rows * cols, dtype=int)
    pos = 0
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                perm[pos] = grid[r, c]
                pos += 1
    return perm


def diagonal_interleave(bits: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    n = rows * cols
    pad = (-len(bits)) % n
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    perm = _diagonal_perm(n, rows, cols)
    blocks = padded.reshape(-1, n)
    out = blocks[:, perm].reshape(-1)
    return out


def diagonal_deinterleave(bits: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    n = rows * cols
    pad = (-len(bits)) % n
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    perm = _diagonal_perm(n, rows, cols)
    inv_perm = np.argsort(perm)
    blocks = padded.reshape(-1, n)
    out = blocks[:, inv_perm].reshape(-1)
    return out[: len(bits)]


def _prng_perm(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def pseudo_random_interleave(bits: np.ndarray, block: int = 64, seed: int = 1234) -> np.ndarray:
    pad = (-len(bits)) % block
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    perm = _prng_perm(block, seed)
    blocks = padded.reshape(-1, block)
    out = blocks[:, perm].reshape(-1)
    return out


def pseudo_random_deinterleave(bits: np.ndarray, block: int = 64, seed: int = 1234) -> np.ndarray:
    pad = (-len(bits)) % block
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    perm = _prng_perm(block, seed)
    inv_perm = np.argsort(perm)
    blocks = padded.reshape(-1, block)
    out = blocks[:, inv_perm].reshape(-1)
    return out[: len(bits)]


INTERLEAVERS = {
    "block": (block_interleave, block_deinterleave),
    "convolutional": (convolutional_interleave, convolutional_deinterleave),
    "diagonal": (diagonal_interleave, diagonal_deinterleave),
    "pseudo_random": (pseudo_random_interleave, pseudo_random_deinterleave),
}
