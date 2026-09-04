"""Rate-1/2, K=7 convolutional code, polynomials (171, 133) octal — the
standard NASA/CCSDS constraint length used across the industry.

Encoder flushes with K-1 zero bits so the trellis always ends in the known
zero state — lets the Viterbi decoder traceback from a fixed endpoint
instead of guessing (standard practice, small overhead: 6 bits/frame).
"""
import numpy as np

K = 7
G1 = 0o171
G2 = 0o133
N_STATES = 1 << (K - 1)


def _poly_bit(reg: int, poly: int) -> int:
    return bin(reg & poly).count("1") % 2


def conv_encode(bits: np.ndarray, g1: int = G1, g2: int = G2) -> np.ndarray:
    flushed = np.concatenate([bits, np.zeros(K - 1, dtype=bits.dtype)])
    reg = 0
    out = []
    for b in flushed:
        reg = ((reg << 1) | int(b)) & ((1 << K) - 1)
        out.append(_poly_bit(reg, g1))
        out.append(_poly_bit(reg, g2))
    return np.array(out, dtype=np.uint8)


def _build_trellis(g1: int = G1, g2: int = G2):
    next_state = np.zeros((N_STATES, 2), dtype=np.int32)
    out_bits = np.zeros((N_STATES, 2, 2), dtype=np.int32)
    for s in range(N_STATES):
        for b in (0, 1):
            reg = ((s << 1) | b) & ((1 << K) - 1)
            out_bits[s, b] = (_poly_bit(reg, g1), _poly_bit(reg, g2))
            next_state[s, b] = reg & (N_STATES - 1)
    return next_state, out_bits


def conv_decode(coded_bits: np.ndarray, g1: int = G1, g2: int = G2) -> np.ndarray:
    """Hard-decision Viterbi. Assumes the K-1 zero-flush tail `conv_encode`
    appends, so traceback starts from the known zero end-state."""
    next_state, out_bits = _build_trellis(g1, g2)
    n_syms = len(coded_bits) // 2
    inf = 1 << 20

    path_metric = np.full(N_STATES, inf)
    path_metric[0] = 0
    prev_state = np.zeros((n_syms, N_STATES), dtype=np.int32)
    prev_bit = np.zeros((n_syms, N_STATES), dtype=np.int8)

    for t in range(n_syms):
        r1, r2 = coded_bits[2 * t], coded_bits[2 * t + 1]
        new_metric = np.full(N_STATES, inf)
        new_prev = np.zeros(N_STATES, dtype=np.int32)
        new_bit = np.zeros(N_STATES, dtype=np.int8)
        for s in range(N_STATES):
            pm = path_metric[s]
            if pm >= inf:
                continue
            for b in (0, 1):
                ns = next_state[s, b]
                o1, o2 = out_bits[s, b]
                bm = (o1 != r1) + (o2 != r2)
                m = pm + bm
                if m < new_metric[ns]:
                    new_metric[ns] = m
                    new_prev[ns] = s
                    new_bit[ns] = b
        path_metric = new_metric
        prev_state[t] = new_prev
        prev_bit[t] = new_bit

    state = 0
    bits_out = np.zeros(n_syms, dtype=np.uint8)
    for t in range(n_syms - 1, -1, -1):
        bits_out[t] = prev_bit[t, state]
        state = prev_state[t, state]
    return bits_out[: -(K - 1)]
