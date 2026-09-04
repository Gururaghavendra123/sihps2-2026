"""Hard-decision slicers — exact inverse of synth/modulate.py's PSK/16-QAM
symbol mappings."""
import numpy as np


def psk_hard_decision(symbols: np.ndarray, order: int, boundary_offset: float | None = None) -> np.ndarray:
    """`boundary_offset` sets where the decision regions are centered.
    Raw modulator output (synth/modulate.py) puts symbols at
    2*pi*k/order + pi/order, so decision boundaries need the default
    step/2 offset. A Costas-loop-corrected stream is different: the M-th
    power phase detector's fixed point drives corrected symbols to land on
    plain multiples of 2*pi/order (see demod/receive.py docstring) — pass
    boundary_offset=0.0 for that path, or hard decisions land exactly on
    the boundary and become a coin flip."""
    bps = int(np.log2(order))
    step = 2 * np.pi / order
    if boundary_offset is None:
        boundary_offset = step / 2
    angles = np.angle(symbols) % (2 * np.pi)
    idx = np.round((angles - boundary_offset) / step).astype(int) % order
    bits = np.zeros((len(idx), bps), dtype=np.uint8)
    for b in range(bps):
        bits[:, b] = (idx >> (bps - 1 - b)) & 1
    return bits.reshape(-1)


_QAM16_LEVELS = np.array([-3, -1, 3, 1])
_QAM16_LEVELS_SCALED = _QAM16_LEVELS / np.sqrt(10)


def qam16_hard_decision(symbols: np.ndarray) -> np.ndarray:
    scaled = symbols * np.sqrt(10)
    i_idx = np.argmin(np.abs(scaled.real[:, None] - _QAM16_LEVELS[None, :]), axis=1)
    q_idx = np.argmin(np.abs(scaled.imag[:, None] - _QAM16_LEVELS[None, :]), axis=1)
    sym = (i_idx << 2) | q_idx
    bits = np.zeros((len(sym), 4), dtype=np.uint8)
    for b in range(4):
        bits[:, b] = (sym >> (3 - b)) & 1
    return bits.reshape(-1)


def nearest_qam16_point(y: complex) -> complex:
    i = _QAM16_LEVELS_SCALED[np.argmin(np.abs(y.real - _QAM16_LEVELS_SCALED))]
    q = _QAM16_LEVELS_SCALED[np.argmin(np.abs(y.imag - _QAM16_LEVELS_SCALED))]
    return complex(i, q)


def rotate_qam16_bits(bits: np.ndarray, rotation: int) -> np.ndarray:
    """Apply a 90*rotation-degree rotation to a hard-decision 16-QAM bit
    stream — used to brute-force the decision-directed loop's residual
    4-fold rotation ambiguity (same real property PSK's Costas loop has;
    resolved for real in phase 4 via CRC/sync-word check)."""
    n_syms = len(bits) // 4
    trimmed = bits[: n_syms * 4].reshape(n_syms, 4)
    i_idx = (trimmed[:, 0] << 1) | trimmed[:, 1]
    q_idx = (trimmed[:, 2] << 1) | trimmed[:, 3]
    z = (_QAM16_LEVELS[i_idx] + 1j * _QAM16_LEVELS[q_idx]) * (1j ** rotation)
    i_idx2 = np.argmin(np.abs(z.real[:, None] - _QAM16_LEVELS[None, :]), axis=1)
    q_idx2 = np.argmin(np.abs(z.imag[:, None] - _QAM16_LEVELS[None, :]), axis=1)
    out = np.zeros((n_syms, 4), dtype=np.uint8)
    out[:, 0] = (i_idx2 >> 1) & 1
    out[:, 1] = i_idx2 & 1
    out[:, 2] = (q_idx2 >> 1) & 1
    out[:, 3] = q_idx2 & 1
    return out.reshape(-1)
