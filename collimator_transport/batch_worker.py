"""
Single-batch worker. Called as a subprocess by run.py.

Usage (not meant to be called directly):
    python -m collimator_transport.batch_worker --n-primaries 1000000 --seed 42 --batch-id 0 --output-dir output
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'collimator_transport' is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-primaries", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    from collimator_transport.main import run_batch

    run_batch(
        n_primaries=args.n_primaries,
        seed=args.seed,
        batch_id=args.batch_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
