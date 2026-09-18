"""Generate Dataset B benchmark samples for the Adaptive Search Engine.

Usage:
    python gen_dataset_b_samples.py          # writes to data/
    python gen_dataset_b_samples.py outdir   # writes to outdir/
"""
import sys
from pathlib import Path

from sigid.synth.dataset_b import BENCHMARK_SCENARIOS, generate_and_save_dataset_b


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    print(f"Generating Dataset B samples → {out_dir.resolve()}/\n")

    for name, params in BENCHMARK_SCENARIOS.items():
        iq_path, gt_path = generate_and_save_dataset_b(params, out_dir, f"datasetB_{name}")
        print(f"  ✓ {name}")
        print(f"    {params.scenario_name}")
        print(f"    {iq_path.name}  +  {gt_path.name}\n")

    print(f"Done — {len(BENCHMARK_SCENARIOS)} scenarios generated.")


if __name__ == "__main__":
    main()
