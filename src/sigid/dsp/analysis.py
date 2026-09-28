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


def extract_eye_diagram(iq: np.ndarray, sps: int, trace_symbols: int = 2, max_traces: int = 40) -> dict:
    """Extract overlaid eye diagram traces for I and Q channels across symbol periods."""
    if sps < 2 or len(iq) < sps * trace_symbols:
        return {"traces_i": [], "traces_q": [], "t": []}
    trace_len = int(sps * trace_symbols)
    available_traces = (len(iq) - sps) // trace_len
    if available_traces <= 0:
        return {"traces_i": [], "traces_q": [], "t": []}
    n_traces = min(max_traces, available_traces)
    t = np.linspace(0, trace_symbols, trace_len).tolist()
    traces_i = []
    traces_q = []
    step = max(1, (len(iq) - trace_len) // n_traces)
    for idx in range(0, len(iq) - trace_len, step):
        chunk = iq[idx : idx + trace_len]
        traces_i.append(chunk.real.tolist())
        traces_q.append(chunk.imag.tolist())
        if len(traces_i) >= n_traces:
            break
    return {"traces_i": traces_i, "traces_q": traces_q, "t": t}


def apply_dsp_filter(
    iq: np.ndarray,
    fs: float,
    filter_type: str = "bypass",
    cutoff_hz: float = 20000.0,
    cutoff_high_hz: float = 60000.0,
    order: int = 4,
    eq_gains: list[float] | None = None,
) -> np.ndarray:
    """Apply DSP digital filtering (Lowpass, Highpass, Bandpass) or 10-band Equalization to IQ stream."""
    if filter_type == "bypass" or len(iq) < 32:
        return iq

    nyq = 0.5 * fs
    if filter_type == "lowpass":
        cutoff = min(max(cutoff_hz, 100.0), nyq * 0.95)
        sos = signal.butter(order, cutoff / nyq, btype="lowpass", output="sos")
        real_filt = signal.sosfilt(sos, iq.real)
        imag_filt = signal.sosfilt(sos, iq.imag)
        return real_filt + 1j * imag_filt
    elif filter_type == "highpass":
        cutoff = min(max(cutoff_hz, 100.0), nyq * 0.95)
        sos = signal.butter(order, cutoff / nyq, btype="highpass", output="sos")
        real_filt = signal.sosfilt(sos, iq.real)
        imag_filt = signal.sosfilt(sos, iq.imag)
        return real_filt + 1j * imag_filt
    elif filter_type == "bandpass":
        c_low = min(max(cutoff_hz, 100.0), nyq * 0.9)
        c_high = min(max(cutoff_high_hz, c_low + 500.0), nyq * 0.98)
        sos = signal.butter(order, [c_low / nyq, c_high / nyq], btype="bandpass", output="sos")
        real_filt = signal.sosfilt(sos, iq.real)
        imag_filt = signal.sosfilt(sos, iq.imag)
        return real_filt + 1j * imag_filt
    elif filter_type == "equalizer" and eq_gains is not None and len(eq_gains) == 10:
        n = len(iq)
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        spec = np.fft.fft(iq)
        center_freqs = [70, 180, 320, 600, 1000, 3000, 6000, 12000, 14000, 16000]
        abs_freqs = np.abs(freqs)
        gains_linear = np.power(10.0, np.array(eq_gains, dtype=float) / 20.0)
        gain_profile = np.interp(abs_freqs, center_freqs, gains_linear, left=gains_linear[0], right=gains_linear[-1])
        shaped_spec = spec * gain_profile
        return np.fft.ifft(shaped_spec)
    return iq

