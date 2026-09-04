"""FSK discriminator demod: per-symbol average instantaneous frequency,
matched to the nearest expected tone — exact inverse of
synth/modulate.py's fsk_modulate."""
import numpy as np


def fsk_demod(iq: np.ndarray, order: int, sps: int, fs: float, tone_spacing: float) -> np.ndarray:
    bps = int(np.log2(order))
    inst_freq = np.diff(np.unwrap(np.angle(iq))) * fs / (2 * np.pi)
    inst_freq = np.concatenate([inst_freq, inst_freq[-1:]])

    tones = (np.arange(order) - (order - 1) / 2) * tone_spacing
    n_symbols = len(iq) // sps

    bits = np.zeros((n_symbols, bps), dtype=np.uint8)
    for k in range(n_symbols):
        seg = inst_freq[k * sps:(k + 1) * sps]
        trim = sps // 4
        core = seg[trim: sps - trim] if sps >= 4 else seg
        avg_freq = np.mean(core) if len(core) > 0 else np.mean(seg)
        idx = int(np.argmin(np.abs(tones - avg_freq)))
        for b in range(bps):
            bits[k, b] = (idx >> (bps - 1 - b)) & 1
    return bits.reshape(-1)
