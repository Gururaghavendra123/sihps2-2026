"""Channel impairments: AWGN + carrier frequency offset."""
import numpy as np


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(iq) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    return iq + noise


def add_freq_offset(iq: np.ndarray, offset_hz: float, fs: float) -> np.ndarray:
    t = np.arange(len(iq)) / fs
    return iq * np.exp(1j * 2 * np.pi * offset_hz * t)
