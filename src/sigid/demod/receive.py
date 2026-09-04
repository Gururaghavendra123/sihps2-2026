"""Full demod chain: raw IQ -> recovered bits. Timing recovery (M&M) and
carrier recovery (Costas) for PSK/QAM; discriminator for FSK.

Phase ambiguity: the M-th power Costas loop locks phase to one of `order`
rotational states (a known property of this class of PLL — it strips the
modulation to find phase, so it can't tell 0 rad from 2*pi/order rad).
Real receivers resolve this via a known preamble, differential encoding, or
— what this project does — the Hypothesis Search Engine's CRC/sync-word
check in phase 4. This module returns bits *up to that ambiguity*; tests
resolve it by brute-forcing the `order` rotations and checking which one's
CRC/BER is clean, exactly mirroring what phase 4 does for real.
"""
import numpy as np

from ..synth.modulate import _PSK_ORDERS, _FSK_ORDERS
from .timing import mm_timing_recovery
from .carrier import costas_loop_psk, costas_loop_decision_directed
from .psk_qam import psk_hard_decision, qam16_hard_decision, nearest_qam16_point
from .fsk import fsk_demod


def demod(iq: np.ndarray, scheme: str, sps: int, fs: float, tone_spacing: float | None = None) -> np.ndarray:
    scheme = scheme.lower()

    if scheme in _PSK_ORDERS:
        order = _PSK_ORDERS[scheme]
        symbols = mm_timing_recovery(iq, sps)
        corrected = costas_loop_psk(symbols, order=order, loop_bw=0.1)
        return psk_hard_decision(corrected, order, boundary_offset=0.0)

    if scheme == "16qam":
        symbols = mm_timing_recovery(iq, sps)
        corrected = costas_loop_decision_directed(symbols, nearest_qam16_point, loop_bw=0.1)
        return qam16_hard_decision(corrected)

    if scheme in _FSK_ORDERS:
        order = _FSK_ORDERS[scheme]
        if tone_spacing is None:
            tone_spacing = fs / sps
        return fsk_demod(iq, order, sps, fs, tone_spacing)

    raise ValueError(f"unknown modulation scheme: {scheme}")


def rotate_psk_bits(bits: np.ndarray, order: int, rotation: int) -> np.ndarray:
    """Apply a constant symbol-index rotation to a hard-decision PSK bit
    stream — used to brute-force the Costas phase ambiguity."""
    bps = int(np.log2(order))
    n_syms = len(bits) // bps
    trimmed = bits[: n_syms * bps].reshape(n_syms, bps)
    weights = 1 << np.arange(bps - 1, -1, -1)
    idx = trimmed @ weights
    rotated_idx = (idx + rotation) % order
    out = np.zeros((n_syms, bps), dtype=np.uint8)
    for b in range(bps):
        out[:, b] = (rotated_idx >> (bps - 1 - b)) & 1
    return out.reshape(-1)
