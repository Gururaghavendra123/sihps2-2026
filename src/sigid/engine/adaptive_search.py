"""Adaptive Candidate Search Engine — 4-stage coarse-to-fine pipeline that
replaces flat 96-hypothesis iteration with intelligent pruning.

Stages:
  1. Physical feature extraction → top-K modulation pre-classification
     (dsp/features.py).  Prunes 50–66% of modulations before demod.
  2. Coarse CFO compensation → derotate baseband IQ so Costas loop
     starts from a better initial condition.
  3. Sync-word gatekeeper → if correlation < threshold after demod,
     skip all 16 FEC×interleaver branches for that modulation.
  4. Priority-queue FEC decoding with early exit → decode in order of
     computational cost (Viterbi → RS → Concatenated → LDPC); stop the
     moment CRC-16 validates AND re-encode BER < threshold.

Returns the same (results, summary) shape as the exhaustive engine
(engine/search.py) so both front ends can render either identically,
plus a list of CandidateTelemetry for the pruning evidence display.

The exhaustive engine (search.py) is preserved unchanged — this is a
parallel, opt-in alternative.
"""
import numpy as np

from ..dsp.features import (
    extract_signal_features,
    classify_top_k_modulations,
    estimate_cfo_coarse,
    correct_cfo,
)
from ..synth.interleavers import INTERLEAVERS
from ..synth.crc import verify_crc16
from ..fec import FEC_ENCODERS, FEC_DECODERS
from ..demod.receive import demod
from ..demod.receive_chain import find_frame_start
from .candidate import CandidateTelemetry, AdaptiveSummary
from .search import _re_encode_ber, _score, CRC_WEIGHT, CORR_WEIGHT, BER_WEIGHT

# FEC decode order: cheapest first.  Viterbi is a single trellis pass;
# RS is algebraic (fast); concatenated chains them; LDPC is iterative
# belief propagation (slowest by far).
FEC_PRIORITY = ["viterbi", "reed_solomon", "concatenated", "ldpc"]

INTERLEAVER_NAMES = list(INTERLEAVERS)

ALL_MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]


def search_adaptive(
    iq: np.ndarray,
    sps: int,
    fs: float,
    n_payload_bits: int,
    top_k: int = 3,
    sync_threshold: float = 0.30,
    ber_threshold: float = 0.02,
    max_shift_bits: int = 140,
) -> tuple[list, dict, list[CandidateTelemetry], AdaptiveSummary]:
    """Adaptive 4-stage search.

    Parameters
    ----------
    iq : complex IQ samples
    sps : samples per symbol
    fs : sample rate (Hz)
    n_payload_bits : expected payload length (bits)
    top_k : how many modulations to keep from the classifier (default 3)
    sync_threshold : minimum sync correlation to proceed (default 0.30)
    ber_threshold : max BER for early-exit acceptance (default 0.02)
    max_shift_bits : search window for sync-word correlation

    Returns
    -------
    results : list of hypothesis dicts (same shape as search.py)
    summary : winner summary dict (same shape as search.py)
    telemetry : list of CandidateTelemetry (one per modulation evaluated)
    adaptive_summary : aggregate pruning statistics
    """
    # ── Stage 1: Coarse CFO estimation & compensation ──────────────────
    # Derotate raw IQ first so phase rotation doesn't wash out higher-order
    # cumulants (C20, C40) during feature extraction.
    cfo_hz = estimate_cfo_coarse(iq, fs)
    iq_corrected = correct_cfo(iq, cfo_hz, fs)

    # ── Stage 2: Feature extraction + modulation pre-classification ────
    features = extract_signal_features(iq_corrected)
    ranked = classify_top_k_modulations(features, k=top_k)
    top_mods = [mod for mod, _score_val in ranked]
    pruned_mods = [m for m in ALL_MODULATIONS if m not in top_mods]

    # ── Stage 3 + 4: Per-modulation demod → gate → priority FEC ────────
    tone_spacing = fs / sps
    n_frame_bits = n_payload_bits + 16  # payload + CRC-16

    all_results = []
    telemetry = []
    total_decodes = 0
    early_exit_triggered = False
    early_exit_winner = None

    for rank_idx, (mod, mod_score) in enumerate(ranked):
        telem = CandidateTelemetry(
            modulation=mod,
            mod_rank=rank_idx + 1,
            mod_score=mod_score,
            cfo_hz=cfo_hz,
        )

        # Demodulate this modulation candidate
        try:
            demod_bits = demod(iq_corrected, mod, sps, fs, tone_spacing=tone_spacing)
        except Exception:
            telem.sync_pruned = True
            telemetry.append(telem)
            continue

        # Sync-word correlation (shared across all FEC/interleaver for this mod)
        shift, rotation, frame_bits, correlation = find_frame_start(
            demod_bits, mod, max_shift_bits
        )
        telem.sync_correlation = correlation

        # ── Stage 3: Gatekeeper ────────────────────────────────────────
        if shift is None or correlation < sync_threshold:
            telem.sync_pruned = True
            telemetry.append(telem)
            continue

        # ── Stage 4: Priority-queue FEC decoding with early exit ───────
        for fec in FEC_PRIORITY:
            if early_exit_triggered:
                break
            for il_name in INTERLEAVER_NAMES:
                _, deinterleave_fn = INTERLEAVERS[il_name]
                entry = {
                    "modulation": mod,
                    "fec": fec,
                    "interleaver": il_name,
                    "correlation": correlation,
                }
                telem.fec_attempted.append(fec)
                telem.interleavers_attempted.append(il_name)
                total_decodes += 1
                telem.decodes_run += 1

                try:
                    deinterleaved = deinterleave_fn(frame_bits)
                    framed_bits = FEC_DECODERS[fec](deinterleaved, n_frame_bits)
                    crc_ok = bool(verify_crc16(framed_bits))
                    ber = _re_encode_ber(fec, il_name, framed_bits, frame_bits)
                    entry.update(
                        crc_ok=crc_ok,
                        ber=ber,
                        payload=framed_bits[:n_payload_bits],
                        error=None,
                    )
                except Exception as exc:
                    entry.update(crc_ok=False, ber=1.0, payload=None, error=str(exc))

                entry["score"] = _score(entry["crc_ok"], correlation, entry["ber"])
                all_results.append(entry)

                # Early exit check
                if entry["crc_ok"] and entry["ber"] < ber_threshold:
                    early_exit_triggered = True
                    early_exit_winner = entry
                    telem.early_exit = True
                    telem.winner_fec = fec
                    telem.winner_interleaver = il_name
                    break

        telemetry.append(telem)
        if early_exit_triggered:
            break

    # ── Build output ───────────────────────────────────────────────────
    all_results.sort(key=lambda e: e["score"], reverse=True)

    if not all_results:
        summary = {
            "status": "no_sync",
            "message": "sync word not found for any candidate modulation (adaptive)",
        }
    elif all_results[0]["crc_ok"]:
        winner = all_results[0]
        summary = {
            "status": "verified",
            "modulation": winner["modulation"],
            "fec": winner["fec"],
            "interleaver": winner["interleaver"],
            "score": winner["score"],
            "payload": winner["payload"],
        }
    else:
        summary = {
            "status": "unknown",
            "message": "no candidate's CRC validated (adaptive)",
            "best_score": all_results[0]["score"],
        }

    # Aggregate telemetry
    mods_pruned_sync = [t.modulation for t in telemetry if t.sync_pruned]
    adaptive_summary = AdaptiveSummary(
        total_candidates_evaluated=len([t for t in telemetry if not t.sync_pruned]),
        total_decodes_run=total_decodes,
        total_decodes_possible=len(ALL_MODULATIONS) * len(FEC_PRIORITY) * len(INTERLEAVER_NAMES),
        modulations_pruned_by_classifier=pruned_mods,
        modulations_pruned_by_sync=mods_pruned_sync,
        early_exit_triggered=early_exit_triggered,
        search_space_reduction_pct=(
            100.0 * (1.0 - total_decodes / (len(ALL_MODULATIONS) * len(FEC_PRIORITY) * len(INTERLEAVER_NAMES)))
            if total_decodes > 0 else 0.0
        ),
    )

    return all_results, summary, telemetry, adaptive_summary
