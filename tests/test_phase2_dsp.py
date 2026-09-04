"""Phase-2 milestone check: DSP estimates validated against synthetic
ground truth. Symbol-rate estimator must work blind (no modulation known),
matching the pipeline order in Final Plan sec 4."""
import numpy as np
import pytest

from sigid.synth.generator import GenParams, generate
from sigid.dsp.analysis import (
    compute_psd,
    compute_spectrogram,
    estimate_bandwidth,
    estimate_snr,
    estimate_symbol_rate,
    extract_constellation,
)

MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]


@pytest.mark.parametrize("modulation", MODULATIONS)
@pytest.mark.parametrize("snr_db", [25, 15, 10])
def test_symbol_rate_estimation_accurate(modulation, snr_db):
    params = GenParams(
        n_payload_bits=4096, modulation=modulation, fec="viterbi",
        interleaver="block", snr_db=snr_db, sps=4, fs=200_000.0,
    )
    iq, _ = generate(params)
    true_symbol_rate = params.fs / params.sps
    est = estimate_symbol_rate(iq, params.fs)
    err_pct = 100 * abs(est - true_symbol_rate) / true_symbol_rate
    assert err_pct < 10, f"{modulation} @ {snr_db}dB: est={est} true={true_symbol_rate} err={err_pct:.1f}%"


def test_psd_and_spectrogram_shapes():
    params = GenParams(n_payload_bits=2048, modulation="qpsk", fec="viterbi", interleaver="block", fs=200_000.0)
    iq, _ = generate(params)
    freqs, psd_db = compute_psd(iq, params.fs)
    assert len(freqs) == len(psd_db)
    assert np.all(np.isfinite(psd_db))

    sxx_freqs, times, sxx_db = compute_spectrogram(iq, params.fs)
    assert sxx_db.shape[0] == len(sxx_freqs)
    assert sxx_db.shape[1] == len(times)
    assert np.all(np.isfinite(sxx_db))


def test_bandwidth_and_snr_estimates_reasonable():
    params = GenParams(n_payload_bits=4096, modulation="qpsk", fec="viterbi", interleaver="block", snr_db=20, sps=4, fs=200_000.0)
    iq, _ = generate(params)
    bw = estimate_bandwidth(iq, params.fs)
    assert 0 < bw < params.fs
    snr = estimate_snr(iq, params.fs)
    assert snr > 0


def test_constellation_extraction_symbol_count():
    params = GenParams(n_payload_bits=4096, modulation="qpsk", fec="viterbi", interleaver="block", snr_db=25, sps=4, fs=200_000.0)
    iq, gt = generate(params)
    symbol_rate = params.fs / params.sps
    expected_symbols = len(iq) / params.sps
    points = extract_constellation(iq, params.fs, symbol_rate, max_symbols=int(expected_symbols) + 10)
    assert abs(len(points) - expected_symbols) / expected_symbols < 0.05
    assert np.all(np.isfinite(points))


def test_constellation_qpsk_four_clusters():
    """At high SNR, QPSK constellation points should cluster near the 4
    ideal phases — sanity check that decimation lands on symbol centers."""
    params = GenParams(n_payload_bits=4096, modulation="qpsk", fec="viterbi", interleaver="block", snr_db=30, sps=4, fs=200_000.0)
    iq, _ = generate(params)
    symbol_rate = params.fs / params.sps
    points = extract_constellation(iq, params.fs, symbol_rate)
    angles = np.angle(points) % (2 * np.pi)
    ideal = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    dists = np.abs(angles[:, None] - ideal[None, :])
    min_dists = np.min(np.minimum(dists, 2 * np.pi - dists), axis=1)
    assert np.mean(min_dists) < 0.3  # radians, tight clustering expected
