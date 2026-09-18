"""Dataset B: Enhanced synthetic signal generator with multipath channel
and prominent CFO, producing stage-by-stage ground truth vectors.

Extends synth/generator.py's GenParams with:
  - multipath: two-ray model (direct + delayed attenuated copy)
  - prominent CFO as a first-class test parameter
  - full intermediate-stage vectors in the ground truth JSON

Used by gen_dataset_b_samples.py to produce the 4 benchmark scenarios
from the adaptive_vs_exhaustive_report.
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
from .generator import get_settle_bits, get_sync_word, SETTLE_BITS, SYNC_WORD_BITS, PREAMBLE_BITS, save_iq, save_wav


@dataclass
class DatasetBParams:
    """Extended generation parameters for Dataset B scenarios."""
    n_payload_bits: int = 512
    modulation: str = "qpsk"
    fec: str = "viterbi"
    interleaver: str = "block"
    sps: int = 4
    fs: float = 200_000.0
    snr_db: float = 20.0
    freq_offset_hz: float = 100.0   # prominent CFO (default higher than GenParams)
    multipath_delay_samples: int = 0  # 0 = no multipath
    multipath_attenuation: float = 0.3  # amplitude of delayed ray (0–1)
    seed: int = 42
    scenario_name: str = "default"


def add_multipath(iq: np.ndarray, delay_samples: int, attenuation: float) -> np.ndarray:
    """Simple two-ray multipath: direct path + delayed attenuated copy."""
    if delay_samples <= 0 or attenuation <= 0:
        return iq
    delayed = np.zeros_like(iq)
    delayed[delay_samples:] = iq[:-delay_samples] * attenuation
    return iq + delayed


def generate_dataset_b(params: DatasetBParams) -> tuple[np.ndarray, dict]:
    """Generate a signal with the full Dataset B pipeline, returning
    stage-by-stage ground truth."""
    rng = np.random.default_rng(params.seed)

    payload_bits = rng.integers(0, 2, params.n_payload_bits, dtype=np.uint8)
    framed_bits = append_crc16(payload_bits)

    if params.fec not in FEC_ENCODERS:
        raise ValueError(f"FEC '{params.fec}' not available")
    fec_bits = FEC_ENCODERS[params.fec](framed_bits)

    interleave_fn, _ = INTERLEAVERS[params.interleaver]
    interleaved_bits = interleave_fn(fec_bits)

    tx_bits = np.concatenate([get_settle_bits(), get_sync_word(), interleaved_bits])

    symbol_rate = params.fs / params.sps
    iq_clean = modulate(tx_bits, params.modulation, params.sps, fs=params.fs, tone_spacing=symbol_rate)

    # Stage-by-stage impairments
    iq_cfo = add_freq_offset(iq_clean, params.freq_offset_hz, params.fs)
    iq_multipath = add_multipath(iq_cfo, params.multipath_delay_samples, params.multipath_attenuation)
    iq_final = add_awgn(iq_multipath, params.snr_db, rng)

    ground_truth = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "B",
        "scenario": params.scenario_name,
        "params": asdict(params),
        "n_preamble_bits": int(PREAMBLE_BITS),
        "n_payload_bits": int(len(payload_bits)),
        "n_framed_bits": int(len(framed_bits)),
        "n_fec_bits": int(len(fec_bits)),
        "n_interleaved_bits": int(len(interleaved_bits)),
        "n_samples": int(len(iq_final)),
        "payload_bits": payload_bits.tolist(),
        "stages": {
            "clean_iq_power": float(np.mean(np.abs(iq_clean) ** 2)),
            "cfo_applied_hz": params.freq_offset_hz,
            "multipath_delay": params.multipath_delay_samples,
            "multipath_atten": params.multipath_attenuation,
            "snr_db": params.snr_db,
        },
    }
    return iq_final, ground_truth


def generate_and_save_dataset_b(params: DatasetBParams, out_dir: Path, name: str) -> tuple[Path, Path]:
    """Generate and save Dataset B signal with all file formats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iq, ground_truth = generate_dataset_b(params)

    iq_path = out_dir / f"{name}.iq"
    wav_path = out_dir / f"{name}.wav"
    gt_path = out_dir / f"{name}.json"

    save_iq(iq, iq_path)
    save_wav(iq, wav_path, params.fs)
    ground_truth["fs"] = params.fs
    gt_path.write_text(json.dumps(ground_truth, indent=2))

    return iq_path, gt_path


# ── Pre-defined benchmark scenarios from the report ────────────────────

BENCHMARK_SCENARIOS = {
    "qpsk_viterbi_block_cfo": DatasetBParams(
        n_payload_bits=512, modulation="qpsk", fec="viterbi", interleaver="block",
        snr_db=20.0, freq_offset_hz=150.0, sps=4, fs=200_000.0,
        scenario_name="QPSK + Viterbi + Block (AWGN + CFO)",
    ),
    "8psk_rs_diagonal_phase": DatasetBParams(
        n_payload_bits=512, modulation="8psk", fec="reed_solomon", interleaver="diagonal",
        snr_db=22.0, freq_offset_hz=80.0, sps=4, fs=200_000.0,
        scenario_name="8-PSK + RS + Diagonal (Phase Offset)",
    ),
    "16qam_ldpc_pseudo_random": DatasetBParams(
        n_payload_bits=512, modulation="16qam", fec="ldpc", interleaver="pseudo_random",
        snr_db=25.0, freq_offset_hz=50.0, multipath_delay_samples=3,
        multipath_attenuation=0.15, sps=4, fs=200_000.0,
        scenario_name="16-QAM + LDPC + Pseudo-random (Multipath)",
    ),
    "2fsk_concat_conv": DatasetBParams(
        n_payload_bits=512, modulation="2fsk", fec="concatenated", interleaver="convolutional",
        snr_db=20.0, freq_offset_hz=120.0, sps=4, fs=200_000.0,
        scenario_name="2-FSK + Concatenated + Conv (Continuous Phase)",
    ),
}
