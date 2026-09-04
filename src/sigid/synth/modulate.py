"""Baseband modulators for the PS candidate set: BPSK, QPSK, 8PSK, 2-FSK,
4-FSK, 16-QAM. Output is complex baseband IQ at `sps` samples/symbol."""
import numpy as np

_PSK_ORDERS = {"bpsk": 2, "qpsk": 4, "8psk": 8}
_FSK_ORDERS = {"2fsk": 2, "4fsk": 4}


def _bits_to_symbols(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    n = len(bits) - (len(bits) % bits_per_symbol)
    bits = bits[:n]
    groups = bits.reshape(-1, bits_per_symbol)
    weights = 1 << np.arange(bits_per_symbol - 1, -1, -1)
    return groups @ weights


def psk_modulate(bits: np.ndarray, order: int, sps: int) -> np.ndarray:
    bps = int(np.log2(order))
    symbols = _bits_to_symbols(bits, bps)
    phases = 2 * np.pi * symbols / order + np.pi / order
    iq = np.exp(1j * phases)
    return np.repeat(iq, sps)


def qam16_modulate(bits: np.ndarray, sps: int) -> np.ndarray:
    symbols = _bits_to_symbols(bits, 4)
    i_bits = (symbols >> 2) & 0b11
    q_bits = symbols & 0b11
    levels = np.array([-3, -1, 3, 1])  # gray-coded mapping
    i = levels[i_bits]
    q = levels[q_bits]
    iq = (i + 1j * q) / np.sqrt(10)
    return np.repeat(iq, sps)


def fsk_modulate(bits: np.ndarray, order: int, sps: int, fs: float, tone_spacing: float) -> np.ndarray:
    bps = int(np.log2(order))
    symbols = _bits_to_symbols(bits, bps)
    freqs = (symbols - (order - 1) / 2) * tone_spacing
    t = np.arange(sps) / fs
    out = np.concatenate([np.exp(1j * 2 * np.pi * f * t) for f in freqs])
    return out


def modulate(bits: np.ndarray, scheme: str, sps: int, fs: float = 1.0, tone_spacing: float | None = None) -> np.ndarray:
    scheme = scheme.lower()
    if scheme in _PSK_ORDERS:
        return psk_modulate(bits, _PSK_ORDERS[scheme], sps)
    if scheme == "16qam":
        return qam16_modulate(bits, sps)
    if scheme in _FSK_ORDERS:
        if tone_spacing is None:
            tone_spacing = fs / sps  # default: one symbol-rate's worth of separation
        return fsk_modulate(bits, _FSK_ORDERS[scheme], sps, fs, tone_spacing)
    raise ValueError(f"unknown modulation scheme: {scheme}")


MODULATIONS = ["bpsk", "qpsk", "8psk", "2fsk", "4fsk", "16qam"]
