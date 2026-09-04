"""Phase-4 milestone check (Final Plan sec 9, Day 4): feed a synthetic file
with an interleaver/FEC combo UNKNOWN to the tool and confirm the
Hypothesis Search Engine finds and proves the right one via CRC, exactly
mirroring what a real analyst session does when the transmitted scheme
isn't told to it in advance (only the modulation, from phase-2 param
estimation, is assumed known — search scope is FEC x interleaver, see
Final Plan sec 5)."""
import numpy as np
import pytest

from sigid.synth.generator import GenParams, generate
from sigid.engine.search import (
    search_hypotheses, search_all_modulations, confidence_label, FECS, INTERLEAVER_NAMES,
)

FEC_INTERLEAVER_COMBOS = [(fec, il) for fec in FECS for il in INTERLEAVER_NAMES]


@pytest.mark.parametrize("fec,interleaver", FEC_INTERLEAVER_COMBOS)
def test_search_finds_correct_combo_among_all_16(fec, interleaver):
    p = GenParams(
        n_payload_bits=256, modulation="qpsk", fec=fec, interleaver=interleaver,
        snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=300.0,
    )
    iq, _ = generate(p)
    rng = np.random.default_rng(p.seed)
    true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

    results, summary = search_hypotheses(
        iq, "qpsk", p.sps, p.fs, p.n_payload_bits, tone_spacing=p.fs / p.sps,
    )

    assert summary["status"] == "verified", f"never verified: {summary}"
    assert summary["fec"] == fec
    assert summary["interleaver"] == interleaver
    assert np.array_equal(summary["payload"], true_payload)
    assert confidence_label(summary) == "VERIFIED"

    # winner must actually be ranked first (highest score) among all 16
    assert results[0]["fec"] == fec and results[0]["interleaver"] == interleaver
    assert results[0]["ber"] == pytest.approx(0.0, abs=1e-9)
    assert len(results) == len(FEC_INTERLEAVER_COMBOS)
    # NOTE: concatenated = conv(RS(msg)), so a plain "viterbi" hypothesis
    # also strips the inner conv layer correctly and can spuriously pass
    # CRC (systematic RS puts the message bits first) when the true fec is
    # concatenated — re-encode BER is what breaks that tie, not CRC alone.


@pytest.mark.parametrize(
    "modulation,fec,interleaver",
    [
        ("qpsk", "viterbi", "block"),
        ("8psk", "reed_solomon", "diagonal"),
        ("16qam", "ldpc", "pseudo_random"),
        ("4fsk", "concatenated", "convolutional"),
    ],
)
def test_search_all_modulations_finds_unknown_modulation_too(modulation, fec, interleaver):
    """search_all_modulations() closes the gap search_hypotheses() leaves:
    modulation itself is now part of the search, not a given — this is
    the real PS requirement #1 ("identify signal parameters — modulation,
    FEC, interleaving") as automatic search, not an analyst-picked
    dropdown."""
    p = GenParams(
        n_payload_bits=256, modulation=modulation, fec=fec, interleaver=interleaver,
        snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=300.0,
    )
    iq, _ = generate(p)
    rng = np.random.default_rng(p.seed)
    true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

    results, summary = search_all_modulations(iq, p.sps, p.fs, p.n_payload_bits)

    assert summary["status"] == "verified", f"never verified: {summary}"
    assert summary["modulation"] == modulation
    assert summary["fec"] == fec
    assert summary["interleaver"] == interleaver
    assert np.array_equal(summary["payload"], true_payload)
    assert confidence_label(summary) == "VERIFIED"
    # wrong-modulation candidates mostly fail to sync at all (empty results
    # for that modulation), so this won't hit the full 96 — just confirm
    # the true modulation contributed its 16 and the winner sits above them all
    assert results[0]["modulation"] == modulation


def test_search_all_modulations_2fsk_bpsk_ambiguity_is_known():
    """Real identifiability gap, not a bug: at the default tone_spacing =
    symbol_rate convention (synth/generator.py), 2-FSK is Sunde's FSK
    (h=1) — each symbol sweeps exactly +-pi of phase, which a plain BPSK
    Costas+hard-decision demod also happens to decode correctly. So a
    2-FSK capture produces byte-identical demod output, and therefore an
    exact score tie, under BOTH the "bpsk" and "2fsk" hypotheses — the
    engine cannot use CRC/correlation/BER to tell them apart, because both
    genuinely reproduce the exact same bits. It still recovers the exact
    right PAYLOAD either way; only the *modulation label* is ambiguous.
    Documented in the landing page as a known limitation rather than
    silently claiming clean modulation ID here. (4-FSK does not have this
    problem — see test_search_all_modulations_finds_unknown_modulation_too.)
    """
    p = GenParams(
        n_payload_bits=256, modulation="2fsk", fec="concatenated", interleaver="convolutional",
        snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=300.0,
    )
    iq, _ = generate(p)
    rng = np.random.default_rng(p.seed)
    true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

    results, summary = search_all_modulations(iq, p.sps, p.fs, p.n_payload_bits)

    assert summary["status"] == "verified"
    assert summary["modulation"] in ("bpsk", "2fsk")
    assert np.array_equal(summary["payload"], true_payload)

    bpsk_entry = next(r for r in results if r["modulation"] == "bpsk" and r["fec"] == "concatenated" and r["interleaver"] == "convolutional")
    fsk_entry = next(r for r in results if r["modulation"] == "2fsk" and r["fec"] == "concatenated" and r["interleaver"] == "convolutional")
    assert bpsk_entry["crc_ok"] and fsk_entry["crc_ok"]
    assert bpsk_entry["score"] == fsk_entry["score"] == 100.0


def test_search_reports_unknown_when_no_sync():
    """Pure noise, no frame at all: engine must say so rather than picking
    a confident-looking wrong answer (Final Plan sec 7 item 4)."""
    rng = np.random.default_rng(0)
    noise_iq = (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)).astype(np.complex128)

    results, summary = search_hypotheses(noise_iq, "qpsk", 4, 200_000.0, 256, tone_spacing=50_000.0)

    assert summary["status"] in ("no_sync", "unknown")
    assert confidence_label(summary) in ("LOW", "HIGH")
    if summary["status"] == "unknown":
        assert all(not r["crc_ok"] for r in results)
