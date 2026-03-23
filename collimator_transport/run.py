"""
Run multiple simulation batches in parallel using subprocess.

Each batch:
  - Runs as a separate Python process (no import/pickling issues).
  - Gets a unique, deterministic seed (base_seed + batch_id).
  - Writes ROOT files to its own output/batch_XXXX/ directory.
  - Keeps file sizes manageable by splitting work across batches.

Usage:
    python -m collimator_transport.run                     # defaults
    python -m collimator_transport.run --total 10000000 --batches 10 --workers 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _run_one_batch(batch_id, n_primaries, seed, output_dir):
    """Run a single batch as a subprocess calling the batch_worker script."""
    cmd = [
        sys.executable, "-m", "collimator_transport.batch_worker",
        "--n-primaries", str(n_primaries),
        "--seed", str(seed),
        "--batch-id", str(batch_id),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[batch {batch_id}] FAILED:\n{result.stderr}")
        return None
    print(f"[batch {batch_id}] done (seed={seed}, n={n_primaries})")
    return str(Path(output_dir) / f"batch_{batch_id:04d}")


def main():
    parser = argparse.ArgumentParser(description="Run SPECT collimator simulation in parallel batches")
    parser.add_argument("--total", type=int, default=1_000_000, help="Total number of primaries")
    parser.add_argument("--batches", type=int, default=4, help="Number of batches to split into")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: min(batches, 4))")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed (batch i gets base_seed + i)")
    parser.add_argument("--output-dir", type=str, default="output", help="Top-level output directory")
    args = parser.parse_args()

    n_per_batch = args.total // args.batches
    remainder = args.total % args.batches
    workers = args.workers or min(args.batches, 4)

    print(f"Total primaries:  {args.total}")
    print(f"Batches:          {args.batches}")
    print(f"Primaries/batch:  {n_per_batch} (+ {remainder} extra in last batch)")
    print(f"Workers:          {workers}")
    print(f"Base seed:        {args.base_seed}")
    print(f"Output dir:       {args.output_dir}")
    print()

    t0 = time.time()

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(args.batches):
            n = n_per_batch + (remainder if i == args.batches - 1 else 0)
            seed = args.base_seed + i
            futures.append(executor.submit(_run_one_batch, i, n, seed, args.output_dir))

        results = []
        for f in as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - t0
    valid = [r for r in results if r is not None]
    print(f"\n{len(valid)}/{args.batches} batches completed in {elapsed:.1f}s")
    print(f"Output directories: {sorted(valid)}")


if __name__ == "__main__":
    main()
