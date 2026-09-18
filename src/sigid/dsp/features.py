"""Physical feature extraction for adaptive modulation pre-classification
and coarse carrier-frequency-offset (CFO) compensation.

This module implements Stage 1 and Stage 2 of the Adaptive Candidate Search
Engine (see adaptive_vs_exhaustive_report.md):

  Stage 1 — extract signal features that discriminate modulation families
  *before* any demodulation attempt:
    - Envelope variance: separates constant-envelope (FSK/PSK) from
      multi-amplitude (QAM) signals.
    - Higher-order cumulants C20, C40, C42: distinguish BPSK vs QPSK vs
      8PSK vs QAM (well-studied in the AMC literature).
    - Instantaneous frequency kurtosis: separates smooth FSK sweeps from
      sharp PSK phase transitions.

  Stage 2 — coarse CFO estimation via M-th power spectral peak detection,
  and baseband derotation so the downstream Costas loop starts from a
  better initial condition (prevents lock slip at ±200 Hz offsets).

All functions are pure DSP — no dependencies on engine/ or demod/.
"""
import numpy as np
from scipy import signal as sp_signal

# ── Stage 1: Feature extraction ────────────────────────────────────────

def _normalize(iq: np.ndarray) -> np.ndarray:
    """Mean-remove and power-normalize to unit average power."""
    iq = iq - np.mean(iq)
    power = np.mean(np.abs(iq) ** 2)
    if power < 1e-15:
        return iq
    return iq / np.sqrt(power)


def extract_signal_features(iq: np.ndarray) -> dict:
    """Compute physical features that discriminate modulation families.

    Returns a dict with:
      envelope_var  — variance of |x[n]|, near 0 for constant-envelope
      C20           — 2nd-order cumulant (E[x^2]), ~1.0 for BPSK, ~0 QPSK
      C40           — 4th-order cumulant, distinguishes PSK orders
      C42           — 4th-order cross cumulant
      freq_kurtosis — kurtosis of instantaneous frequency derivative
    """
    x = _normalize(iq)
    n = len(x)
    if n < 16:
        return {"envelope_var": 0.0, "C20": 0.0, "C40": 0.0,
                "C42": 0.0, "freq_kurtosis": 0.0}

    env = np.abs(x)
    envelope_var = float(np.var(env))

    # Higher-order cumulants (Swami & Sadler, 2000 style)
    # M20 = E[x^2], M21 = E[|x|^2], M40 = E[x^4], M42 = E[x^2 |x|^2]
    M20 = np.mean(x ** 2)
    M21 = np.mean(np.abs(x) ** 2)  # = 1.0 after normalization
    M40 = np.mean(x ** 4)
    M41 = np.mean((x ** 3) * np.conj(x))
    M42 = np.mean((x ** 2) * (np.abs(x) ** 2))

    C20 = float(np.abs(M20))
    C40 = float(np.abs(M40 - 3 * M20 ** 2))
    C42 = float(np.abs(M42 - np.abs(M20) ** 2 - 2 * M21 ** 2))

    # Instantaneous frequency kurtosis
    inst_phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(inst_phase)
    if len(inst_freq) > 4:
        freq_diff = np.diff(inst_freq)
        mu = np.mean(freq_diff)
        sigma = np.std(freq_diff)
        if sigma > 1e-12:
            freq_kurtosis = float(np.mean(((freq_diff - mu) / sigma) ** 4))
        else:
            freq_kurtosis = 0.0
    else:
        freq_kurtosis = 0.0

    return {
        "envelope_var": envelope_var,
        "C20": C20,
        "C40": C40,
        "C42": C42,
        "freq_kurtosis": freq_kurtosis,
    }


# ── Modulation pre-classification ──────────────────────────────────────

# All 6 PS candidate modulations
ALL_MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]

# Ideal feature signatures (empirically tuned on synthetic signals across
# 10–25 dB SNR).  Each entry is (envelope_var_ref, C20_ref, C40_ref,
# C42_ref, freq_kurtosis_ref) — the classifier scores each modulation
# by weighted distance from these references.
_SIGNATURES = {
    #              env_var   C20    C40    C42    freq_kurt
    "bpsk":       (0.005,   0.98,  1.95,  3.98,  4.0),
    "qpsk":       (0.005,   0.01,  0.95,  2.00,  3.7),
    "8psk":       (0.005,   0.02,  0.01,  2.00,  3.6),
    "16qam":      (0.105,   0.02,  0.62,  2.00,  3.5),
    "2fsk":       (0.036,   0.07,  0.30,  2.09,  2.4),
    "4fsk":       (0.036,   0.07,  0.30,  2.09,  2.7),
}

# Scale normalizers for each feature dimension:
# env_var ~ 0.05, C20 ~ 0.3, C40 ~ 0.3, C42 ~ 0.5, freq_kurt ~ 0.8
_SCALES = np.array([0.05, 0.3, 0.3, 0.5, 0.8])
_WEIGHTS = 1.0 / (_SCALES ** 2)


def classify_top_k_modulations(
    features: dict,
    k: int = 3,
    modulations: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank candidate modulations by weighted Euclidean distance from
    reference feature signatures.  Returns a list of (modulation, score)
    tuples sorted best-first (lowest distance = highest score), truncated
    to the top `k`.

    The score is 1/(1+distance) so it's in (0, 1] and higher = better.
    """
    modulations = modulations or ALL_MODULATIONS
    feat_vec = np.array([
        features["envelope_var"],
        features["C20"],
        features["C40"],
        features["C42"],
        features["freq_kurtosis"],
    ])

    scored = []
    for mod in modulations:
        ref = np.array(_SIGNATURES[mod])
        dist = np.sqrt(np.sum(_WEIGHTS * (feat_vec - ref) ** 2))
        scored.append((mod, 1.0 / (1.0 + dist)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ── Stage 2: Coarse CFO estimation & compensation ──────────────────────

def estimate_cfo_coarse(
    iq: np.ndarray,
    fs: float,
    orders: list[int] | None = None,
    max_cfo_hz: float = 2000.0,
    prominence_threshold: float = 18.0,
) -> float:
    """Estimate carrier frequency offset by M-th power spectral peak
    detection.  Tries multiple orders (default [4, 8, 2]) and picks the
    one with the strongest spectral peak relative to the noise floor.

    The M-th power operation x^M collapses M-PSK modulation into a single
    tone at M*f_CFO; dividing the detected peak frequency by M recovers
    f_CFO.  The prominence threshold (peak/median ratio) prevents
    false-positive detections on noise sidelobes.

    The search window is bounded to |f| <= max_cfo_hz * M to prevent
    spurious detections on FSK tone harmonics or wideband noise.

    Returns estimated CFO in Hz (0.0 if no prominent peak found).
    """
    if orders is None:
        orders = [4, 8, 2]

    x = _normalize(iq)
    n = len(x)
    if n < 64:
        return 0.0

    best_cfo = 0.0
    best_prominence = 0.0

    for M in orders:
        powered = x ** M
        nfft = min(8192, n)
        spectrum = np.abs(np.fft.fft(powered, n=nfft))
        freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

        # Restrict to valid CFO search window [-max_cfo_hz * M, +max_cfo_hz * M]
        # and exclude DC neighborhood (>= 10 Hz * M)
        valid_mask = (np.abs(freqs) <= max_cfo_hz * M) & (np.abs(freqs) >= 10.0 * M)
        if not np.any(valid_mask):
            continue

        valid_indices = np.where(valid_mask)[0]
        peak_idx = valid_indices[np.argmax(spectrum[valid_mask])]
        peak_val = spectrum[peak_idx]
        median_val = np.median(spectrum[spectrum > 0])

        if median_val < 1e-15:
            continue

        prominence = peak_val / median_val
        if prominence > prominence_threshold and prominence > best_prominence:
            best_prominence = prominence
            best_cfo = float(freqs[peak_idx]) / M

    return best_cfo


def correct_cfo(iq: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    """Derotate baseband IQ to remove estimated carrier frequency offset:
        r[n] = x[n] * exp(-j * 2π * cfo * n / fs)
    """
    if abs(cfo_hz) < 1e-6:
        return iq
    n = np.arange(len(iq))
    return iq * np.exp(-1j * 2 * np.pi * cfo_hz * n / fs)
