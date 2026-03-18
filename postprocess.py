"""
Post-processing: match incoming and outgoing photons from phase-space ROOT files.

Strategy (optimized for large incoming files):
  1. Load the SMALL outgoing file first.
  2. Extract the set of (EventID, TrackID) pairs we need from outgoing.
  3. Stream through the LARGE incoming file in chunks, keeping only rows
     that match an outgoing event (+ a sample of non-matched for context).
  4. Build the final numpy array.

Output: a numpy array where each row is one incoming photon, with columns:
  [in_x, in_y, in_theta, in_phi, in_E,
   out_x, out_y, out_theta, out_phi, out_E]

Outgoing columns are zero for incoming photons with no outgoing match.

Usage:
    python postprocess.py
"""

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


def postprocess(
    incoming_path="phsp/collimator_incoming.root",
    outgoing_path="phsp/collimator_outgoing.root",
    incoming_tree="ps_incoming",
    outgoing_tree="ps_outgoing",
    chunk_size=500_000,
):
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load outgoing (small file, ~MB)
    # ------------------------------------------------------------------
    out_path = Path(outgoing_path)
    has_outgoing = out_path.exists() and out_path.stat().st_size > 0

    if has_outgoing:
        print(f"Loading outgoing ({out_path.stat().st_size / 1e6:.1f} MB) ...")
        f_out = uproot.open(outgoing_path)
        out = f_out[outgoing_tree].arrays(OUTGOING_BRANCHES, library="numpy")
        n_out = len(out["EventID"])
        print(f"  {n_out} outgoing photons loaded in {time.time() - t0:.1f}s")

        out_evt = out["EventID"].astype(np.int64)
        out_trk = out["TrackID"].astype(np.int64)
        out_par = out["ParentID"].astype(np.int64)

        # Keys we need to find in incoming:
        # direct match keys (EventID, TrackID) and parent match keys (EventID, ParentID)
        needed_keys = set(out_evt * PRIME + out_trk) | set(out_evt * PRIME + out_par)
    else:
        print("No outgoing file found.")
        n_out = 0
        needed_keys = set()

    # ------------------------------------------------------------------
    # 2. Stream through incoming in chunks — collect ALL rows
    # ------------------------------------------------------------------
    t1 = time.time()
    in_file = uproot.open(incoming_path)
    in_tree = in_file[incoming_tree]
    n_in_total = in_tree.num_entries
    print(f"Incoming file: {n_in_total} entries, streaming in chunks of {chunk_size} ...")

    all_in_chunks = []
    total_loaded = 0

    for chunk in in_tree.iterate(INCOMING_BRANCHES, library="numpy", step_size=chunk_size):
        n_chunk = len(chunk["EventID"])
        total_loaded += n_chunk
        all_in_chunks.append(chunk)

        if total_loaded % (chunk_size * 10) == 0 or total_loaded == n_in_total:
            elapsed = time.time() - t1
            pct = total_loaded / n_in_total * 100
            print(f"  Loaded {total_loaded}/{n_in_total} ({pct:.0f}%) in {elapsed:.1f}s")

    # Merge all chunks
    print("Merging chunks ...")
    inc = {key: np.concatenate([c[key] for c in all_in_chunks]) for key in INCOMING_BRANCHES}
    del all_in_chunks
    n_in = len(inc["EventID"])
    print(f"  {n_in} incoming photons merged in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 3. Build result array
    # ------------------------------------------------------------------
    t2 = time.time()
    print("Building result array ...")
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

    # ------------------------------------------------------------------
    # 4. Match outgoing → incoming (vectorized where possible)
    # ------------------------------------------------------------------
    if has_outgoing and n_out > 0:
        print("Matching outgoing to incoming ...")

        inc_evt = inc["EventID"].astype(np.int64)
        inc_trk = inc["TrackID"].astype(np.int64)
        inc_keys = inc_evt * PRIME + inc_trk

        # Build lookup: key -> row index (last occurrence wins if duplicates)
        inc_key_to_row = dict(zip(inc_keys.tolist(), range(n_in)))

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
        print(f"  {n_matched} outgoing matched to incoming")

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
    else:
        n_matched = 0

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
    print(f"\nFirst 5 rows (all):")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(result[:5])

    # Show some matched rows
    matched_mask = result[:, COL_OUT_E] > 0
    if matched_mask.any():
        print(f"\nFirst 5 rows with outgoing photon:")
        print(result[matched_mask][:5])


if __name__ == "__main__":
    t_start = time.time()
    result = postprocess()

    out_file = Path("postprocessed_data.npy")
    np.save(out_file, result)
    print(f"\nSaved to {out_file} ({result.nbytes / 1e6:.1f} MB)")
    print(f"Total post-processing time: {time.time() - t_start:.1f}s")
