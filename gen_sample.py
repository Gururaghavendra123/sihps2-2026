"""CLI: generate one sample synthetic file into data/ for manual GUI testing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sigid.synth.generator import GenParams, generate_and_save

if __name__ == "__main__":
    params = GenParams(
        n_payload_bits=4096,
        modulation="qpsk",
        fec="viterbi",
        interleaver="block",
        snr_db=15,
        fs=200_000.0,
    )
    out_dir = Path(__file__).resolve().parent / "data"
    iq_path, gt_path = generate_and_save(params, out_dir, "sample_qpsk_viterbi_block")
    print(f"wrote {iq_path}")
    print(f"wrote {gt_path}")
    print(f"wrote {iq_path.with_suffix('.wav')}")
