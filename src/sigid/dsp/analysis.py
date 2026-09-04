"""Core DSP analysis: FFT, PSD, spectrogram, bandwidth/SNR/symbol-rate
estimation, constellation extraction (Final Plan sec 9, Day 2)."""
import numpy as np
from scipy import signal


def compute_fft(iq: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(iq)
    spectrum = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    mag_db = 20 * np.log10(np.abs(spectrum) / n + 1e-12)
    return freqs, mag_db


def compute_psd(iq: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(iq, fs=fs, nperseg=min(nperseg, len(iq)), return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-15)
    return freqs, psd_db


def compute_spectrogram(iq: np.ndarray, fs: float, nperseg: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, sxx = signal.spectrogram(
        iq, fs=fs, nperseg=min(nperseg, len(iq)), return_onesided=False
    )
    freqs = np.fft.fftshift(freqs)
    sxx = np.fft.fftshift(sxx, axes=0)
    sxx_db = 10 * np.log10(sxx + 1e-15)
    return freqs, times, sxx_db


def estimate_bandwidth(iq: np.ndarray, fs: float, threshold_db: float = -20.0) -> float:
    freqs, psd_db = compute_psd(iq, fs)
    peak = np.max(psd_db)
    above = freqs[psd_db > peak + threshold_db]
    if len(above) == 0:
        return 0.0
    return float(above.max() - above.min())


def estimate_snr(iq: np.ndarray, fs: float) -> float:
    freqs, psd_db = compute_psd(iq, fs)
    peak = np.max(psd_db)
    noise_floor = np.median(psd_db[psd_db < peak - 20])
    return float(peak - noise_floor)


def estimate_symbol_rate(iq: np.ndarray, fs: float, nperseg: int = 4096) -> float:
    """Modulation-agnostic symbol-clock recovery (Oerder & Meyr style
    nonlinearity timing estimator), run BEFORE modulation is known — matches
    the pipeline order in Final Plan sec 4. Tries two nonlinear features:

      - amplitude-jump: |x[n+1]-x[n]|^2 — strong for PSK/QAM (non-constant
        envelope, rectangular pulses -> abrupt jumps at symbol edges)
      - freq-jump: |d(inst_freq)/dn|^2 — catches constant-envelope FSK where
        the amplitude feature is comparatively weak

    Both candidates' PSDs are computed; the one with the higher peak
    prominence (peak / median) wins. Empirically 0% error across all 6 PS
    candidate modulations at 10-25 dB SNR on rectangular-pulse synthetic
    signals (see phase-2 prototyping).
    """
    if len(iq) < 8:
        return 0.0

    feat_amp = np.abs(np.diff(iq)) ** 2
    inst_freq = np.diff(np.unwrap(np.angle(iq)))
    feat_freq = np.abs(np.diff(inst_freq)) ** 2

    best_prominence = -1.0
    best_freq = 0.0
    for feat in (feat_amp, feat_freq):
        n = min(nperseg, len(feat))
        if n < 16:
            continue
        freqs, psd = signal.welch(feat, fs=fs, nperseg=n)
        lo = max(2, len(psd) // 200)
        if lo >= len(psd):
            continue
        idx = np.argmax(psd[lo:]) + lo
        prominence = psd[idx] / (np.median(psd[lo:]) + 1e-15)
        if prominence > best_prominence:
            best_prominence = prominence
            best_freq = float(freqs[idx])
    return abs(best_freq)


def extract_constellation(iq: np.ndarray, fs: float, symbol_rate: float, max_symbols: int = 2000) -> np.ndarray:
    """Decimate at the recovered symbol clock to pull one IQ sample per
    symbol. Rectangular pulses (no ISI in our synthetic generator) mean any
    phase offset within the symbol lands on a valid point, so this skips
    fine timing-phase search — good enough for phase-2 visualization."""
    if symbol_rate <= 0:
        return np.array([], dtype=complex)
    sps = fs / symbol_rate
    if sps < 1:
        return np.array([], dtype=complex)
    indices = np.round(np.arange(0, len(iq), sps)).astype(int)
    indices = indices[indices < len(iq)]
    symbols = iq[indices]
    if len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
    return symbols
