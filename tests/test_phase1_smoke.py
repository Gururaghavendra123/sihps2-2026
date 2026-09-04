"""Phase-1 milestone check: generate a synthetic file, load it back, run FFT.
Also round-trips every interleaver and confirms CRC-16 validates on a clean
(no-noise) frame — the exact signal the Hypothesis Search Engine relies on."""
import numpy as np
import pytest

from sigid.synth.generator import GenParams, generate, generate_and_save
from sigid.synth.interleavers import INTERLEAVERS
from sigid.synth.crc import append_crc16, verify_crc16
from sigid.io.loader import load
from sigid.dsp.analysis import compute_fft, estimate_bandwidth


def test_crc16_roundtrip():
    bits = np.random.default_rng(0).integers(0, 2, 500, dtype=np.uint8)
    framed = append_crc16(bits)
    assert verify_crc16(framed)
    framed[10] ^= 1  # corrupt one bit
    assert not verify_crc16(framed)


@pytest.mark.parametrize("name", list(INTERLEAVERS.keys()))
def test_interleaver_roundtrip(name):
    fn, inv = INTERLEAVERS[name]
    bits = np.random.default_rng(1).integers(0, 2, 256, dtype=np.uint8)
    out = inv(fn(bits))
    assert np.array_equal(out, bits)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "8psk", "2fsk", "4fsk", "16qam"])
@pytest.mark.parametrize("fec", ["viterbi", "reed_solomon", "concatenated", "ldpc"])
def test_generate_all_combinations(modulation, fec):
    params = GenParams(n_payload_bits=512, modulation=modulation, fec=fec, interleaver="block", snr_db=20)
    iq, gt = generate(params)
    assert len(iq) > 0
    assert np.all(np.isfinite(iq))


def test_generate_load_fft_roundtrip(tmp_path):
    params = GenParams(n_payload_bits=2048, modulation="qpsk", fec="viterbi", interleaver="block", snr_db=15, fs=200_000.0)
    iq_path, gt_path = generate_and_save(params, tmp_path, "smoke_test")

    assert iq_path.exists()
    assert gt_path.exists()

    loaded = load(iq_path, fs=params.fs)
    assert len(loaded.iq) > 0

    freqs, mag_db = compute_fft(loaded.iq, params.fs)
    assert len(freqs) == len(loaded.iq)
    assert np.all(np.isfinite(mag_db))

    bw = estimate_bandwidth(loaded.iq, params.fs)
    assert bw > 0

    wav_path = iq_path.with_suffix(".wav")
    loaded_wav = load(wav_path)
    assert len(loaded_wav.iq) > 0
