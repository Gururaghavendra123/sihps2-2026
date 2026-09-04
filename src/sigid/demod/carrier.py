"""Carrier phase/frequency recovery, corrects the residual offset left by
`channel.add_freq_offset`. Two variants:

- `costas_loop_psk`: blind M-th power method for constant-modulus M-PSK.
  Raising a symbol to the M-th power cancels the M-ary data phase and
  leaves only the carrier offset — but this only works when every
  constellation point has the same magnitude. 16-QAM's off-diagonal points
  (e.g. the ones at magnitude sqrt(1+9)) don't collapse to a single angle
  under that operation, so the loop doesn't converge for QAM (verified:
  ~33% BER even noise-free). Use `costas_loop_decision_directed` there.

- `costas_loop_decision_directed`: makes a hard decision each symbol
  against the current best constellation guess, then tracks the phase
  error between the received sample and that decision — works for any
  constellation shape, PSK included.
"""
import numpy as np


def costas_loop_psk(symbols: np.ndarray, order: int, loop_bw: float = 0.05, damping: float = 0.707) -> np.ndarray:
    n = len(symbols)
    denom = 1 + 2 * damping * loop_bw + loop_bw ** 2
    alpha = (4 * damping * loop_bw) / denom
    beta = (4 * loop_bw ** 2) / denom

    phase = 0.0
    freq = 0.0
    out = np.zeros(n, dtype=complex)
    for i in range(n):
        y = symbols[i] * np.exp(-1j * phase)
        out[i] = y
        err = np.angle(y ** order) / order if abs(y) > 1e-12 else 0.0
        freq += beta * err
        phase += freq + alpha * err
    return out


def costas_loop_decision_directed(symbols: np.ndarray, nearest_point_fn, loop_bw: float = 0.05, damping: float = 0.707) -> np.ndarray:
    """`nearest_point_fn(complex) -> complex` snaps a sample to the nearest
    ideal constellation point (see psk_qam.nearest_qam16_point)."""
    n = len(symbols)
    denom = 1 + 2 * damping * loop_bw + loop_bw ** 2
    alpha = (4 * damping * loop_bw) / denom
    beta = (4 * loop_bw ** 2) / denom

    phase = 0.0
    freq = 0.0
    out = np.zeros(n, dtype=complex)
    for i in range(n):
        y = symbols[i] * np.exp(-1j * phase)
        out[i] = y
        d = nearest_point_fn(y)
        err = (y * np.conj(d)).imag / (abs(d) ** 2 + 1e-12)
        freq += beta * err
        phase += freq + alpha * err
    return out
