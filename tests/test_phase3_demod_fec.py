"""Phase-3 milestone check (Final Plan sec 9, Day 3): every one of the 8
algorithms (6 modulations' demod, 4 FEC decoders — well, both together —
plus all 4 interleavers) proven correct against ground truth, standalone,
independent of the GUI.

Two ambiguities are inherent to the algorithms themselves, not bugs:
  - Costas/decision-directed carrier loops lock to one of several
    rotational fixed points (see demod/receive.py docstring)
  - loops need a short acquisition transient before locking, which is why
    synth/generator.py transmits a throwaway settle region + sync word
    ahead of every frame (see demod/receive_chain.py docstring)
A real receiver resolves both via sync-word correlation, which is what
demod/receive_chain.py's receive_known_chain() does — this is the same
mechanism Final Plan sec 5's Hypothesis Search Engine will reuse in phase 4
to additionally search over unknown FEC/interleaver combinations.
"""
import numpy as np
import pytest

from sigid.synth.generator import GenParams, generate
from sigid.synth.crc import append_crc16
from sigid.fec import FEC_ENCODERS, FEC_DECODERS
from sigid.demod.receive_chain import receive_known_chain

MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]
FECS = ["viterbi", "reed_solomon", "concatenated", "ldpc"]
INTERLEAVERS = ["block", "convolutional", "diagonal", "pseudo_random"]


@pytest.mark.parametrize("fec", FECS)
def test_fec_decoder_corrects_bit_errors(fec):
    """Standalone FEC round-trip: encode, flip a few bits, decode, confirm
    exact recovery — independent of demod/GUI."""
    rng = np.random.default_rng(7)
    payload_bits = rng.integers(0, 2, 256, dtype=np.uint8)
    framed = append_crc16(payload_bits)
    coded = FEC_ENCODERS[fec](framed)

    noisy = coded.copy()
    n_flips = 2
    flip_idx = rng.choice(len(noisy), size=n_flips, replace=False)
    noisy[flip_idx] ^= 1

    decoded = FEC_DECODERS[fec](noisy, len(framed))
    assert np.array_equal(decoded, framed)


@pytest.mark.parametrize("interleaver", INTERLEAVERS)
def test_receive_chain_all_interleavers(interleaver):
    p = GenParams(
        n_payload_bits=256, modulation="qpsk", fec="viterbi", interleaver=interleaver,
        snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=300.0,
    )
    iq, _ = generate(p)
    rng = np.random.default_rng(p.seed)
    true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

    recovered, evidence = receive_known_chain(
        iq, "qpsk", "viterbi", interleaver, p.sps, p.fs, p.n_payload_bits,
        tone_spacing=p.fs / p.sps,
    )
    assert recovered is not None, f"CRC never validated: {evidence}"
    assert np.array_equal(recovered, true_payload)


@pytest.mark.parametrize("modulation", MODULATIONS)
@pytest.mark.parametrize("fec", FECS)
def test_receive_chain_all_modulation_fec_combos(modulation, fec):
    """The real phase-3 milestone: feed a synthetic file through demod ->
    sync -> deinterleave -> FEC decode and recover the EXACT original
    payload bits, for every modulation x FEC combination the PS names."""
    p = GenParams(
        n_payload_bits=256, modulation=modulation, fec=fec, interleaver="block",
        snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=300.0,
    )
    iq, _ = generate(p)
    rng = np.random.default_rng(p.seed)
    true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

    recovered, evidence = receive_known_chain(
        iq, modulation, fec, "block", p.sps, p.fs, p.n_payload_bits,
        tone_spacing=p.fs / p.sps,
    )
    assert recovered is not None, f"CRC never validated: {evidence}"
    assert np.array_equal(recovered, true_payload)
