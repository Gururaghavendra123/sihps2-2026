"""Mueller & Muller symbol-timing recovery — tracks the correct sampling
instant against an unknown initial timing offset, operating directly on the
oversampled complex baseband (linear interpolation between raw samples for
the fractional part)."""
import numpy as np


def _interp(iq: np.ndarray, pos: float) -> complex:
    i0 = int(np.floor(pos))
    frac = pos - i0
    if i0 + 1 >= len(iq):
        return iq[-1]
    return iq[i0] * (1 - frac) + iq[i0 + 1] * frac


def _slice(sample: complex) -> complex:
    return complex(np.sign(sample.real) or 1, np.sign(sample.imag) or 1)


def mm_timing_recovery(iq: np.ndarray, sps: float, gain: float = 0.05) -> np.ndarray:
    """Returns one recovered complex sample per symbol."""
    pos = sps / 2  # start mid-first-symbol, arbitrary initial phase
    symbols = []
    prev_sample = _interp(iq, pos)
    prev_decision = _slice(prev_sample)
    pos += sps

    while pos < len(iq) - 1:
        sample = _interp(iq, pos)
        decision = _slice(sample)

        error = (decision.conjugate() * sample - prev_decision.conjugate() * prev_sample).real
        pos += sps + gain * np.clip(error, -1, 1)

        symbols.append(sample)
        prev_sample, prev_decision = sample, decision

    return np.array(symbols, dtype=complex)
