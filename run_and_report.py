"""
Quick single-batch run + report (for testing).

For parallel multi-batch runs use:
    python -m collimator_transport.run --total 10000000 --batches 10 --workers 4

Usage:
    python run_and_report.py
"""

import sys
from collimator_transport.main import run_batch


def main():
    batch_dir = run_batch(
        n_primaries=1_000_000,
        seed=42,
        batch_id=0,
        output_dir="output",
    )
    print(f"\nDone. Output in: {batch_dir}")
    print("Run 'python postprocess.py' to build the numpy array.")


if __name__ == "__main__":
    sys.exit(main())
