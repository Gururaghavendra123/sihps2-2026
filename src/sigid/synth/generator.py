"""End-to-end synthetic signal generator:
preamble -> bits -> CRC-16 -> FEC(one of 4) -> interleave(one of 4)
     -> modulate -> AWGN + freq offset -> save .iq/.wav + ground-truth JSON.

The preamble (PREAMBLE_BITS throwaway bits, prepended before the real
frame) exists because timing/carrier recovery (demod/timing.py,
demod/carrier.py) has an unavoidable acquisition transient — a real
receiver always burns a short lock-in period before the payload starts;
without one, the first few real data bits get eaten while the loops
converge, and there's no way to recover them after the fact. Its content
doesn't matter, only its length.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .crc import append_crc16
from .interleavers import INTERLEAVERS
from ..fec import FEC_ENCODERS
from .modulate import modulate
from .channel import add_awgn, add_freq_offset


SETTLE_BITS = 100  # pure throwaway — loops are still acquiring during this,
                    # so its content is never correlated against (empirically,
                    # Costas+MM lock within ~30 symbols at 25dB SNR; this
                    # leaves comfortable margin)
SYNC_WORD_BITS = 32  # comes right after settle, once loops have locked —
                      # this is what the receiver actually correlates against


def get_settle_bits() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, SETTLE_BITS, dtype=np.uint8)


def get_sync_word() -> np.ndarray:
    """Fixed, publicly-known pattern (a real sync word is meant to be known
    in advance) — the receiver correlates against this to find frame start,
    rather than blindly trying every alignment against the FEC decoder
    (see demod/receive_chain.py)."""
    return np.random.default_rng(1).integers(0, 2, SYNC_WORD_BITS, dtype=np.uint8)


PREAMBLE_BITS = SETTLE_BITS + SYNC_WORD_BITS


@dataclass
class GenParams:
    n_payload_bits: int = 2048
    modulation: str = "qpsk"
    fec: str = "viterbi"
    interleaver: str = "block"
    sps: int = 4
    fs: float = 200_000.0
    snr_db: float = 15.0
    freq_offset_hz: float = 0.0
    seed: int = 42


def generate(params: GenParams) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(params.seed)

    payload_bits = rng.integers(0, 2, params.n_payload_bits, dtype=np.uint8)
    framed_bits = append_crc16(payload_bits)

    if params.fec not in FEC_ENCODERS:
        raise ValueError(f"FEC '{params.fec}' not available (see fec/ldpc.py stub)")
    fec_bits = FEC_ENCODERS[params.fec](framed_bits)

    interleave_fn, _ = INTERLEAVERS[params.interleaver]
    interleaved_bits = interleave_fn(fec_bits)

    tx_bits = np.concatenate([get_settle_bits(), get_sync_word(), interleaved_bits])

    symbol_rate = params.fs / params.sps
    iq = modulate(tx_bits, params.modulation, params.sps, fs=params.fs, tone_spacing=symbol_rate)
    iq = add_freq_offset(iq, params.freq_offset_hz, params.fs)
    iq = add_awgn(iq, params.snr_db, rng)

    ground_truth = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": asdict(params),
        "n_preamble_bits": int(PREAMBLE_BITS),
        "n_payload_bits": int(len(payload_bits)),
        "n_framed_bits": int(len(framed_bits)),
        "n_fec_bits": int(len(fec_bits)),
        "n_interleaved_bits": int(len(interleaved_bits)),
        "n_samples": int(len(iq)),
        "payload_bits": payload_bits.tolist(),
    }
    return iq, ground_truth


def save_iq(iq: np.ndarray, path: Path) -> None:
    """Raw interleaved float32 I/Q, matching common SDR .iq convention."""
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(path)


def save_wav(iq: np.ndarray, path: Path, fs: float) -> None:
    import wave

    peak = np.max(np.abs(iq)) or 1.0
    i16_i = (iq.real / peak * 32767).astype(np.int16)
    i16_q = (iq.imag / peak * 32767).astype(np.int16)
    stereo = np.empty(2 * len(iq), dtype=np.int16)
    stereo[0::2] = i16_i
    stereo[1::2] = i16_q
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(stereo.tobytes())


def generate_and_save(params: GenParams, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iq, ground_truth = generate(params)

    iq_path = out_dir / f"{name}.iq"
    wav_path = out_dir / f"{name}.wav"
    gt_path = out_dir / f"{name}.json"

    save_iq(iq, iq_path)
    save_wav(iq, wav_path, params.fs)
    ground_truth["fs"] = params.fs
    gt_path.write_text(json.dumps(ground_truth, indent=2))

    return iq_path, gt_path
