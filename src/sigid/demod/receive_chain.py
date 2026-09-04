"""Full known-scheme receive chain: raw IQ -> demod -> preamble correlation
-> deinterleave -> FEC decode -> verified payload bits.

`demod()` recovers bits up to two ambiguities that are properties of the
algorithms themselves, not bugs (see receive.py's docstring for why):
  - an `order`-fold carrier-phase rotation (Costas/decision-directed loop
    lock point), and
  - a small, variable acquisition-transient delay (the loops need a few
    symbols to lock; synth/generator.py prepends a throwaway preamble so
    the real frame survives that transient intact).

A real receiver resolves both by correlating against a known preamble/sync
word — cheap, O(shift x rotation x preamble_len) bit comparisons — THEN
running the (expensive) FEC decoder once at the winning alignment. Trying
every alignment against the FEC decoder directly (brute-forcing there
instead) is the naive version and is combinatorially too slow — correlate
first, decode once.
"""
import numpy as np

from ..synth.modulate import _PSK_ORDERS
from ..synth.interleavers import INTERLEAVERS
from ..synth.crc import verify_crc16
from ..synth.generator import get_sync_word, SYNC_WORD_BITS
from ..fec import FEC_DECODERS
from .receive import demod, rotate_psk_bits
from .psk_qam import rotate_qam16_bits


def _candidate_rotations(bits: np.ndarray, scheme: str):
    if scheme in _PSK_ORDERS:
        order = _PSK_ORDERS[scheme]
        for rot in range(order):
            yield rot, rotate_psk_bits(bits, order, rot)
    elif scheme == "16qam":
        for rot in range(4):
            yield rot, rotate_qam16_bits(bits, rot)
    else:
        yield 0, bits


def find_frame_start(
    demod_bits: np.ndarray, scheme: str, max_shift_bits: int
) -> tuple[int | None, int | None, np.ndarray, float]:
    """Correlate against the known sync word (NOT the settle region before
    it — those bits are still mid-acquisition and unreliable, see module
    docstring) across (shift, rotation) candidates.

    Public (no leading underscore): the Hypothesis Search Engine
    (engine/search.py, phase 4) reuses this directly. Sync-word correlation
    and rotation are resolved once per modulation scheme, independent of
    which FEC/interleaver the engine is trying — the sync word is
    transmitted uncoded, ahead of the FEC+interleave stage (see
    synth/generator.py) — so one call here serves every hypothesis it
    scores.

    Returns (best_shift, best_rotation, bits_from_frame_start, correlation),
    where correlation is the winning candidate's fraction of matching sync
    bits (1.0 = perfect match — the "Sync-word correlation" evidence-panel
    figure)."""
    sync_word = get_sync_word()
    best = (None, None, len(sync_word) + 1)  # (shift, rotation, mismatches)

    for rot, rotated in _candidate_rotations(demod_bits, scheme):
        limit = min(max_shift_bits, len(rotated) - len(sync_word))
        for shift in range(max(0, limit)):
            window = rotated[shift: shift + len(sync_word)]
            mismatches = int(np.sum(window != sync_word))
            if mismatches < best[2]:
                best = (shift, rot, mismatches)

    shift, rotation, mismatches = best
    correlation = 1.0 - mismatches / len(sync_word)
    if shift is None or mismatches > len(sync_word) // 4:
        return None, None, np.array([], dtype=np.uint8), correlation

    _, rotated = next((r, b) for r, b in _candidate_rotations(demod_bits, scheme) if r == rotation)
    return shift, rotation, rotated[shift + SYNC_WORD_BITS:], correlation


def receive_known_chain(
    iq: np.ndarray,
    scheme: str,
    fec: str,
    interleaver: str,
    sps: int,
    fs: float,
    n_payload_bits: int,
    tone_spacing: float | None = None,
    max_shift_bits: int = 140,
):
    """Returns (payload_bits, evidence) — evidence describes how the frame
    was found and whether its CRC validated.

    `max_shift_bits` must cover the transmitted settle region
    (synth/generator.py's SETTLE_BITS) plus margin — that's the acquisition
    transient this search is scanning past to find the sync word."""
    demod_bits = demod(iq, scheme, sps, fs, tone_spacing=tone_spacing)
    _, deinterleave_fn = INTERLEAVERS[interleaver]

    shift, rotation, frame_bits, correlation = find_frame_start(demod_bits, scheme, max_shift_bits)
    if shift is None:
        return None, {"error": "preamble not found", "correlation": correlation}

    try:
        deinterleaved = deinterleave_fn(frame_bits)
        framed_bits = FEC_DECODERS[fec](deinterleaved, n_payload_bits + 16)
    except Exception as exc:
        return None, {"shift": shift, "rotation": rotation, "correlation": correlation, "fec_error": str(exc)}

    crc_ok = verify_crc16(framed_bits)
    evidence = {"shift": shift, "rotation": rotation, "correlation": correlation, "crc_ok": bool(crc_ok)}
    if crc_ok:
        return framed_bits[:n_payload_bits], evidence
    return None, evidence
