"""
Post-processing: match incoming and outgoing photons across all batch directories.

Scans output/batch_*/ for ROOT files, merges them, matches incoming→outgoing,
and produces a single numpy array.

Output columns:
  [in_x, in_y, in_theta, in_phi, in_E,
   out_x, out_y, out_theta, out_phi, out_E]

Usage:
    python postprocess.py                          # default output/ dir
    python postprocess.py --output-dir output      # explicit
"""

import argparse
import numpy as np
import uproot
from pathlib import Path
import time


COL_IN_X, COL_IN_Y, COL_IN_THETA, COL_IN_PHI, COL_IN_E = 0, 1, 2, 3, 4
COL_OUT_X, COL_OUT_Y, COL_OUT_THETA, COL_OUT_PHI, COL_OUT_E = 5, 6, 7, 8, 9
N_COLS = 10

COLUMN_NAMES = [
    "in_x", "in_y", "in_theta", "in_phi", "in_E",
    "out_x", "out_y", "out_theta", "out_phi", "out_E",
]

PRIME = np.int64(1_000_000_007)

INCOMING_BRANCHES = [
    "EventID", "TrackID",
    "PrePosition_X", "PrePosition_Y",
    "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
    "KineticEnergy",
]

OUTGOING_BRANCHES = [
    "EventID", "TrackID", "ParentID",
    "PostPosition_X", "PostPosition_Y",
    "PostDirection_X", "PostDirection_Y", "PostDirection_Z",
    "KineticEnergy",
]


def direction_to_spherical(dx, dy, dz):
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.where(r == 0, 1.0, r)
    theta = np.arccos(np.clip(dz / r, -1.0, 1.0))
    phi = np.arctan2(dy, dx)
    return theta, phi


def find_batch_dirs(output_dir="output"):
    """Find all batch_XXXX directories under output_dir."""
    root = Path(output_dir)
    dirs = sorted(root.glob("batch_*"))
    if not dirs:
        # Fallback: check for legacy phsp/ directory
        if (Path("phsp") / "collimator_incoming.root").exists():
            return [Path(".")]
    return dirs


def load_and_merge_trees(batch_dirs, subpath, tree_name, branches):
    """Load a tree from multiple batch directories and merge into one dict of arrays."""
    all_chunks = {b: [] for b in branches}
    total = 0

    for bd in batch_dirs:
        fpath = bd / subpath
        if not fpath.exists() or fpath.stat().st_size == 0:
            continue
        f = uproot.open(str(fpath))
        tree = f[tree_name]
        arrays = tree.arrays(branches, library="numpy")
        for b in branches:
            all_chunks[b].append(arrays[b])
        total += len(arrays[branches[0]])

    if total == 0:
        return None, 0

    merged = {b: np.concatenate(all_chunks[b]) for b in branches}
    return merged, total


def postprocess(output_dir="output"):
    t0 = time.time()

    batch_dirs = find_batch_dirs(output_dir)
    print(f"Found {len(batch_dirs)} batch directories")

    # Load outgoing (small)
    print("Loading outgoing ROOT files ...")
    out, n_out = load_and_merge_trees(
        batch_dirs, Path("phsp") / "collimator_outgoing.root", "ps_outgoing", OUTGOING_BRANCHES
    )
    print(f"  {n_out} outgoing photons total")

    # Load incoming (large, but streamed per batch)
    print("Loading incoming ROOT files ...")
    inc, n_in = load_and_merge_trees(
        batch_dirs, Path("phsp") / "collimator_incoming.root", "ps_incoming", INCOMING_BRANCHES
    )
    print(f"  {n_in} incoming photons total (loaded in {time.time() - t0:.1f}s)")

    if inc is None or n_in == 0:
        print("No incoming data found.")
        return np.zeros((0, N_COLS))

    # Build result
    keV = 1000.0
    result = np.zeros((n_in, N_COLS), dtype=np.float64)

    result[:, COL_IN_X] = inc["PrePosition_X"]
    result[:, COL_IN_Y] = inc["PrePosition_Y"]
    result[:, COL_IN_E] = inc["KineticEnergy"] * keV

    in_theta, in_phi = direction_to_spherical(
        inc["PreDirection_X"], inc["PreDirection_Y"], inc["PreDirection_Z"]
    )
    result[:, COL_IN_THETA] = in_theta
    result[:, COL_IN_PHI] = in_phi

    # Match outgoing → incoming
    n_matched = 0
    if out is not None and n_out > 0:
        print("Matching outgoing to incoming ...")
        t2 = time.time()

        inc_evt = inc["EventID"].astype(np.int64)
        inc_trk = inc["TrackID"].astype(np.int64)
        inc_keys = inc_evt * PRIME + inc_trk
        inc_key_to_row = dict(zip(inc_keys.tolist(), range(n_in)))

        out_evt = out["EventID"].astype(np.int64)
        out_trk = out["TrackID"].astype(np.int64)
        out_par = out["ParentID"].astype(np.int64)

        out_keys_direct = (out_evt * PRIME + out_trk).tolist()
        out_keys_parent = (out_evt * PRIME + out_par).tolist()

        matched_in_rows = []
        matched_out_idx = []

        for j in range(n_out):
            row = inc_key_to_row.get(out_keys_direct[j], -1)
            if row == -1:
                row = inc_key_to_row.get(out_keys_parent[j], -1)
            if row >= 0:
                matched_in_rows.append(row)
                matched_out_idx.append(j)

        n_matched = len(matched_in_rows)

        if n_matched > 0:
            rows = np.array(matched_in_rows, dtype=np.int64)
            idx = np.array(matched_out_idx, dtype=np.int64)

            out_theta, out_phi = direction_to_spherical(
                out["PostDirection_X"][idx],
                out["PostDirection_Y"][idx],
                out["PostDirection_Z"][idx],
            )

            result[rows, COL_OUT_X] = out["PostPosition_X"][idx]
            result[rows, COL_OUT_Y] = out["PostPosition_Y"][idx]
            result[rows, COL_OUT_THETA] = out_theta
            result[rows, COL_OUT_PHI] = out_phi
            result[rows, COL_OUT_E] = out["KineticEnergy"][idx] * keV

        print(f"  Done in {time.time() - t2:.1f}s")

    _print_summary(result, n_in, n_matched)
    return result


def _print_summary(result, n_in, n_matched):
    n_with_out = np.count_nonzero(result[:, COL_OUT_E])
    print("\n----- Post-processing summary -----")
    print(f"Total incoming photons:       {n_in}")
    print(f"Outgoing matched to incoming: {n_matched}")
    print(f"Rows with nonzero outgoing E: {n_with_out}")
    print(f"Rows with zero outgoing (absorbed/lost): {n_in - n_with_out}")
    if n_in > 0:
        print(f"Transmission fraction: {n_with_out / n_in:.6g}")
    print(f"\nOutput array shape: {result.shape}")
    print(f"Columns: {COLUMN_NAMES}")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(f"\nFirst 5 rows:")
    print(result[:5])
    matched_mask = result[:, COL_OUT_E] > 0
    if matched_mask.any():
        print(f"\nFirst 5 rows with outgoing photon:")
        print(result[matched_mask][:5])


def main():
    parser = argparse.ArgumentParser(description="Post-process SPECT simulation batches")
    parser.add_argument("--output-dir", type=str, default="output", help="Top-level output directory")
    args = parser.parse_args()

    t_start = time.time()
    result = postprocess(output_dir=args.output_dir)

    out_file = Path("postprocessed_data.npy")
    np.save(out_file, result)
    print(f"\nSaved to {out_file} ({result.nbytes / 1e6:.1f} MB)")
    print(f"Total post-processing time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
