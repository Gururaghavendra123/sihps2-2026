"""Hypothesis Search Engine (Final Plan sec 5): search over unknown FEC x
interleaver combinations for a signal whose modulation is already known
(estimated in phase 2 / picked by the analyst), score each by CRC-16
validity, sync-word correlation, and re-encode BER, and rank them for the
evidence panel.

CRC-16 is the bulletproof signal (planted in the frame header specifically
so the engine has a hard pass/fail, not just heuristics). Sync-word
correlation and phase rotation are resolved once per modulation via
demod/receive_chain.find_frame_start() — they don't depend on the
FEC/interleaver guess, since the sync word is transmitted uncoded ahead of
the FEC+interleave stage (see synth/generator.py) — so every hypothesis
below shares that one figure; re-encode BER is what actually varies with
and discriminates the FEC/interleaver guess: re-encoding+re-interleaving a
correct decode reproduces the received coded bits almost exactly (only
real channel errors differ), a wrong guess does not.
"""
import numpy as np

from ..synth.interleavers import INTERLEAVERS
from ..synth.crc import verify_crc16
from ..synth.modulate import _PSK_ORDERS, _FSK_ORDERS
from ..fec import FEC_ENCODERS, FEC_DECODERS
from ..demod.receive import demod
from ..demod.receive_chain import find_frame_start

FECS = list(FEC_ENCODERS)
INTERLEAVER_NAMES = list(INTERLEAVERS)
MODULATIONS = list(_PSK_ORDERS) + ["16qam"] + list(_FSK_ORDERS)

# Evidence-panel weighting (Final Plan sec 5): correlation is weighted
# heaviest since it's shared ground truth about frame alignment; CRC adds a
# bonus for the one candidate that actually decodes; BER is a tiebreak /
# fallback signal when nothing's CRC-clean. Sums to 100 when crc_ok and
# ber == 0.
CRC_WEIGHT = 25.0
CORR_WEIGHT = 60.0
BER_WEIGHT = 15.0


def _re_encode_ber(fec: str, interleaver: str, framed_bits: np.ndarray, received_coded_bits: np.ndarray) -> float:
    """Re-encode+re-interleave a hypothesis's decoded frame and compare
    against the actually-received coded bits — the signal that
    discriminates the right FEC/interleaver guess from a wrong one."""
    encode_fn = FEC_ENCODERS[fec]
    interleave_fn, _ = INTERLEAVERS[interleaver]
    re_coded = interleave_fn(encode_fn(framed_bits))
    n = min(len(re_coded), len(received_coded_bits))
    if n == 0:
        return 1.0
    return float(np.mean(re_coded[:n] != received_coded_bits[:n]))


def _score(crc_ok: bool, correlation: float, ber: float) -> float:
    raw = CRC_WEIGHT * crc_ok + CORR_WEIGHT * correlation + BER_WEIGHT * (1.0 - min(ber, 1.0))
    return float(np.clip(raw, 0.0, 100.0))


def search_hypotheses(
    iq: np.ndarray,
    scheme: str,
    sps: int,
    fs: float,
    n_payload_bits: int,
    tone_spacing: float | None = None,
    max_shift_bits: int = 140,
):
    """Try every (FEC, interleaver) combination against one demodulated
    capture and rank them — the phase-4 counterpart to
    demod/receive_chain.receive_known_chain(), which needs fec/interleaver
    told to it in advance instead of searching for them.

    Returns (results, summary):
      - `results`: every one of the 16 hypotheses, sorted best-score-first,
        each a dict with fec, interleaver, correlation, crc_ok, ber, score,
        payload (recovered bits, or None if decode failed) and error.
      - `summary`: the winner if its CRC validated ("verified"), else a
        report that no candidate proved out ("unknown") — the
        unknown-signal fallback (Final Plan sec 7 item 4): never assert a
        confident answer that isn't CRC-backed.
    """
    demod_bits = demod(iq, scheme, sps, fs, tone_spacing=tone_spacing)
    shift, rotation, frame_bits, correlation = find_frame_start(demod_bits, scheme, max_shift_bits)

    if shift is None:
        return [], {"status": "no_sync", "message": "sync word not found at any shift/rotation", "correlation": correlation}

    n_frame_bits = n_payload_bits + 16
    results = []
    for fec in FECS:
        for interleaver in INTERLEAVER_NAMES:
            _, deinterleave_fn = INTERLEAVERS[interleaver]
            entry = {"fec": fec, "interleaver": interleaver, "correlation": correlation}
            try:
                deinterleaved = deinterleave_fn(frame_bits)
                framed_bits = FEC_DECODERS[fec](deinterleaved, n_frame_bits)
                crc_ok = bool(verify_crc16(framed_bits))
                ber = _re_encode_ber(fec, interleaver, framed_bits, frame_bits)
                entry.update(crc_ok=crc_ok, ber=ber, payload=framed_bits[:n_payload_bits], error=None)
            except Exception as exc:
                entry.update(crc_ok=False, ber=1.0, payload=None, error=str(exc))
            entry["score"] = _score(entry["crc_ok"], correlation, entry["ber"])
            results.append(entry)

    results.sort(key=lambda e: e["score"], reverse=True)
    winner = results[0]
    if winner["crc_ok"]:
        summary = {
            "status": "verified",
            "fec": winner["fec"],
            "interleaver": winner["interleaver"],
            "score": winner["score"],
            "payload": winner["payload"],
        }
    else:
        summary = {"status": "unknown", "message": "no candidate's CRC validated", "best_score": winner["score"]}
    return results, summary


def search_all_modulations(
    iq: np.ndarray,
    sps: int,
    fs: float,
    n_payload_bits: int,
    modulations: list[str] | None = None,
    max_shift_bits: int = 140,
):
    """Full catalog search: modulation x FEC x interleaver (6 x 4 x 4 = 96
    hypotheses), closing the gap search_hypotheses() leaves — that one
    still needs the modulation told to it. sps is NOT modulation-dependent
    (it's the receiver's samples-per-symbol from the recovered symbol
    clock, e.g. dsp/analysis.estimate_symbol_rate — the same for whichever
    modulation guess is under test), so one sps/fs pair drives every
    modulation candidate here, matching PS requirement #1 ("identify
    signal parameters — modulation, FEC, interleaving") as a real
    automatic search rather than an analyst-supplied modulation.

    Returns (results, summary) shaped like search_hypotheses(), with each
    result additionally carrying its "modulation" and results pooled
    across all candidate modulations before ranking.
    """
    modulations = modulations or MODULATIONS
    tone_spacing = fs / sps
    all_results = []
    for mod in modulations:
        try:
            mod_results, _ = search_hypotheses(
                iq, mod, sps, fs, n_payload_bits, tone_spacing=tone_spacing, max_shift_bits=max_shift_bits,
            )
        except Exception:
            mod_results = []
        for r in mod_results:
            r["modulation"] = mod
        all_results.extend(mod_results)

    if not all_results:
        return [], {"status": "no_sync", "message": "sync word not found for any candidate modulation"}

    all_results.sort(key=lambda e: e["score"], reverse=True)
    winner = all_results[0]
    if winner["crc_ok"]:
        summary = {
            "status": "verified",
            "modulation": winner["modulation"],
            "fec": winner["fec"],
            "interleaver": winner["interleaver"],
            "score": winner["score"],
            "payload": winner["payload"],
        }
    else:
        summary = {"status": "unknown", "message": "no candidate's CRC validated across any modulation", "best_score": winner["score"]}
    return all_results, summary


def confidence_label(summary: dict) -> str:
    """VERIFIED / HIGH / LOW labels derived from the score (Final Plan sec
    7 item 3) — never asserted from a raw percentage alone."""
    if summary.get("status") == "verified":
        return "VERIFIED"
    if summary.get("status") == "unknown":
        return "HIGH" if summary.get("best_score", 0) >= 60 else "LOW"
    return "LOW"
