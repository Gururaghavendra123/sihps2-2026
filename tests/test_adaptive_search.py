"""Tests for the Adaptive Candidate Search Engine — verifies feature
discrimination, CFO compensation, modulation pre-classification,
sync gatekeeper pruning, priority-queue early exit, and payload parity
against the exhaustive engine.

Runs against synthetic signals with known ground truth, same as the
existing test_phase4_hypothesis_search.py does for the exhaustive engine.
"""
import numpy as np
import pytest

from sigid.synth.generator import GenParams, generate
from sigid.synth.dataset_b import (
    DatasetBParams,
    generate_dataset_b,
    BENCHMARK_SCENARIOS,
)
from sigid.dsp.features import (
    extract_signal_features,
    classify_top_k_modulations,
    estimate_cfo_coarse,
    correct_cfo,
)
from sigid.engine.adaptive_search import search_adaptive
from sigid.engine.search import search_all_modulations, confidence_label


# ── Feature extraction tests ──────────────────────────────────────────


class TestFeatureExtraction:
    """Verify that physical features discriminate modulation families."""

    @pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "8psk"])
    def test_psk_has_low_envelope_variance(self, modulation):
        """Constant-envelope PSK should have near-zero envelope variance."""
        p = GenParams(n_payload_bits=4096, modulation=modulation, fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)
        features = extract_signal_features(iq)
        assert features["envelope_var"] < 0.10, \
            f"{modulation}: envelope_var={features['envelope_var']:.3f} (expected < 0.10)"

    def test_qam16_has_high_envelope_variance(self):
        """Multi-amplitude 16-QAM should have higher envelope variance."""
        p = GenParams(n_payload_bits=4096, modulation="16qam", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)
        features = extract_signal_features(iq)
        assert features["envelope_var"] > 0.03, \
            f"16qam: envelope_var={features['envelope_var']:.3f} (expected > 0.03)"

    @pytest.mark.parametrize("modulation", ["2fsk", "4fsk"])
    def test_fsk_has_low_freq_kurtosis(self, modulation):
        """FSK's smooth frequency sweeps should have low kurtosis."""
        p = GenParams(n_payload_bits=4096, modulation=modulation, fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)
        features = extract_signal_features(iq)
        # FSK kurtosis should be notably different from PSK
        assert features["freq_kurtosis"] < 20.0, \
            f"{modulation}: freq_kurtosis={features['freq_kurtosis']:.1f}"

    def test_bpsk_has_high_c20(self):
        """BPSK produces strong C20 ~1.0."""
        p = GenParams(n_payload_bits=4096, modulation="bpsk", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)
        features = extract_signal_features(iq)
        assert features["C20"] > 0.3, \
            f"bpsk: C20={features['C20']:.3f} (expected > 0.3)"


# ── CFO compensation tests ────────────────────────────────────────────


class TestCFOCompensation:
    """Verify CFO estimation and derotation."""

    def test_estimates_positive_cfo(self):
        p = GenParams(n_payload_bits=4096, modulation="qpsk", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0,
                      freq_offset_hz=150.0)
        iq, _ = generate(p)
        cfo = estimate_cfo_coarse(iq, p.fs)
        # Should be within 50 Hz of the true 150 Hz offset
        assert abs(abs(cfo) - 150.0) < 50.0, \
            f"CFO estimate {cfo:.1f} Hz, expected ~150 Hz"

    def test_correct_cfo_derotates(self):
        """After derotation, re-estimating CFO should give ~0."""
        p = GenParams(n_payload_bits=4096, modulation="qpsk", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0,
                      freq_offset_hz=100.0)
        iq, _ = generate(p)
        cfo = estimate_cfo_coarse(iq, p.fs)
        corrected = correct_cfo(iq, cfo, p.fs)
        residual = estimate_cfo_coarse(corrected, p.fs)
        assert abs(residual) < 30.0, \
            f"residual CFO after correction: {residual:.1f} Hz"


# ── Modulation pre-classification tests ───────────────────────────────


class TestModulationClassifier:
    """Verify that the true modulation appears in top-k ranking."""

    @pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "8psk", "16qam", "4fsk"])
    def test_true_mod_in_top_3(self, modulation):
        p = GenParams(n_payload_bits=4096, modulation=modulation, fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)
        features = extract_signal_features(iq)
        top3 = classify_top_k_modulations(features, k=3)
        top3_mods = [m for m, _ in top3]
        # 2fsk/bpsk ambiguity is known — either is acceptable for 2fsk
        if modulation == "2fsk":
            assert "2fsk" in top3_mods or "bpsk" in top3_mods, \
                f"neither 2fsk nor bpsk in top-3: {top3_mods}"
        else:
            assert modulation in top3_mods, \
                f"{modulation} not in top-3: {top3_mods}"


# ── Adaptive search engine tests ──────────────────────────────────────


class TestAdaptiveSearch:
    """End-to-end adaptive engine tests — payload parity with exhaustive."""

    @pytest.mark.parametrize("modulation,fec,interleaver", [
        ("qpsk", "viterbi", "block"),
        ("8psk", "reed_solomon", "diagonal"),
        ("4fsk", "concatenated", "convolutional"),
    ])
    def test_adaptive_recovers_correct_payload(self, modulation, fec, interleaver):
        """Adaptive engine must recover byte-identical payload to ground truth."""
        p = GenParams(
            n_payload_bits=256, modulation=modulation, fec=fec, interleaver=interleaver,
            snr_db=25, sps=4, fs=200_000.0, freq_offset_hz=100.0,
        )
        iq, _ = generate(p)
        rng = np.random.default_rng(p.seed)
        true_payload = rng.integers(0, 2, p.n_payload_bits, dtype=np.uint8)

        results, summary, telemetry, adaptive_summary = search_adaptive(
            iq, p.sps, p.fs, p.n_payload_bits,
        )

        assert summary["status"] == "verified", f"not verified: {summary}"
        assert summary["fec"] == fec
        assert summary["interleaver"] == interleaver
        assert np.array_equal(summary["payload"], true_payload)

    @pytest.mark.parametrize("modulation,fec,interleaver", [
        ("qpsk", "viterbi", "block"),
        ("8psk", "reed_solomon", "diagonal"),
    ])
    def test_adaptive_uses_fewer_decodes_than_exhaustive(self, modulation, fec, interleaver):
        """Adaptive engine must attempt fewer decodes than exhaustive (96)."""
        p = GenParams(
            n_payload_bits=256, modulation=modulation, fec=fec, interleaver=interleaver,
            snr_db=25, sps=4, fs=200_000.0,
        )
        iq, _ = generate(p)

        _, _, _, adaptive_summary = search_adaptive(iq, p.sps, p.fs, p.n_payload_bits)

        assert adaptive_summary.total_decodes_run < 96, \
            f"adaptive ran {adaptive_summary.total_decodes_run} decodes (expected < 96)"
        assert adaptive_summary.search_space_reduction_pct > 0, \
            f"no pruning occurred"

    def test_adaptive_prunes_noise(self):
        """Pure noise should be pruned — no verified candidate."""
        rng = np.random.default_rng(0)
        noise_iq = (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)).astype(np.complex128)

        results, summary, telemetry, adaptive_summary = search_adaptive(
            noise_iq, 4, 200_000.0, 256,
        )

        assert summary["status"] in ("no_sync", "unknown")
        # Most modulations should be sync-pruned
        pruned_count = sum(1 for t in telemetry if t.sync_pruned)
        assert pruned_count > 0, "expected some modulations to be sync-pruned on noise"

    def test_telemetry_structure(self):
        """Telemetry objects should have correct structure."""
        p = GenParams(n_payload_bits=256, modulation="qpsk", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)

        _, _, telemetry, adaptive_summary = search_adaptive(iq, p.sps, p.fs, p.n_payload_bits)

        assert len(telemetry) > 0
        for t in telemetry:
            assert hasattr(t, "modulation")
            assert hasattr(t, "mod_rank")
            assert hasattr(t, "sync_pruned")
            assert hasattr(t, "early_exit")
            d = t.to_dict()
            assert isinstance(d, dict)
            assert "modulation" in d

        assert hasattr(adaptive_summary, "total_decodes_run")
        d = adaptive_summary.to_dict()
        assert "search_space_reduction_pct" in d

    def test_early_exit_triggers_on_clean_signal(self):
        """On a clean high-SNR signal, early exit should trigger."""
        p = GenParams(n_payload_bits=256, modulation="qpsk", fec="viterbi",
                      interleaver="block", snr_db=25, sps=4, fs=200_000.0)
        iq, _ = generate(p)

        _, summary, telemetry, adaptive_summary = search_adaptive(iq, p.sps, p.fs, p.n_payload_bits)

        assert summary["status"] == "verified"
        assert adaptive_summary.early_exit_triggered, "early exit should trigger on clean signal"
        winner_telem = [t for t in telemetry if t.early_exit]
        assert len(winner_telem) == 1


# ── Dataset B benchmark tests ─────────────────────────────────────────


class TestDatasetB:
    """Verify adaptive search against Dataset B benchmark scenarios."""

    @pytest.mark.parametrize("scenario_name", [
        "qpsk_viterbi_block_cfo",
        "8psk_rs_diagonal_phase",
        "2fsk_concat_conv",
    ])
    def test_dataset_b_scenario_verified(self, scenario_name):
        """Each Dataset B scenario should produce a verified result."""
        params = BENCHMARK_SCENARIOS[scenario_name]
        iq, gt = generate_dataset_b(params)
        true_payload = np.array(gt["payload_bits"], dtype=np.uint8)

        results, summary, telemetry, adaptive_summary = search_adaptive(
            iq, params.sps, params.fs, params.n_payload_bits,
        )

        assert summary["status"] == "verified", \
            f"scenario {scenario_name}: {summary}"
        assert summary["modulation"] == params.modulation or \
            (params.modulation == "2fsk" and summary["modulation"] in ("2fsk", "bpsk")), \
            f"modulation mismatch: {summary['modulation']} vs {params.modulation}"
        assert summary["fec"] == params.fec
        assert summary["interleaver"] == params.interleaver
        assert np.array_equal(summary["payload"], true_payload)

    def test_dataset_b_16qam_ldpc(self):
        """16-QAM + LDPC + pseudo_random with multipath — harder scenario.
        Uses the exhaustive engine as reference since LDPC at moderate SNR
        with multipath may challenge the adaptive classifier."""
        params = BENCHMARK_SCENARIOS["16qam_ldpc_pseudo_random"]
        iq, gt = generate_dataset_b(params)
        true_payload = np.array(gt["payload_bits"], dtype=np.uint8)

        # Exhaustive reference
        exh_results, exh_summary = search_all_modulations(
            iq, params.sps, params.fs, params.n_payload_bits,
        )

        # Adaptive
        adp_results, adp_summary, _, _ = search_adaptive(
            iq, params.sps, params.fs, params.n_payload_bits,
        )

        # If exhaustive verifies, adaptive should too
        if exh_summary["status"] == "verified":
            assert adp_summary["status"] == "verified", \
                f"exhaustive verified but adaptive didn't: {adp_summary}"
            assert np.array_equal(adp_summary["payload"], exh_summary["payload"])
