"""
Quick inspection of the postprocessed numpy array.

Usage:
    python inspect_data.py
"""

import numpy as np

IN_X, IN_Y, IN_THETA, IN_PHI, IN_E = 0, 1, 2, 3, 4
OUT_X, OUT_Y, OUT_THETA, OUT_PHI, OUT_E = 5, 6, 7, 8, 9


def main():
    data = np.load("postprocessed_data.npy")
    inspect(data)


def inspect(data):
    print(f"Array shape: {data.shape}")
    print(f"Columns: [in_x, in_y, in_theta, in_phi, in_E, out_x, out_y, out_theta, out_phi, out_E]\n")

    transmitted = data[data[:, OUT_E] > 0]
    absorbed = data[data[:, OUT_E] == 0]

    print(f"Total incoming:    {len(data)}")
    print(f"Transmitted:       {len(transmitted)}")
    print(f"Absorbed/lost:     {len(absorbed)}")
    print(f"Transmission:      {len(transmitted) / len(data):.6f}\n")

    print("--- Incoming energy (keV) ---")
    print(f"  Min:  {data[:, IN_E].min():.1f}")
    print(f"  Max:  {data[:, IN_E].max():.1f}")
    print(f"  Mean: {data[:, IN_E].mean():.1f}\n")

    if len(transmitted) > 0:
        print("--- Outgoing energy (keV, transmitted only) ---")
        print(f"  Min:  {transmitted[:, OUT_E].min():.1f}")
        print(f"  Max:  {transmitted[:, OUT_E].max():.1f}")
        print(f"  Mean: {transmitted[:, OUT_E].mean():.1f}\n")

        print("--- First 10 transmitted photons ---")
        np.set_printoptions(precision=3, suppress=True, linewidth=140)
        print(f"{'in_x':>8} {'in_y':>8} {'in_θ':>8} {'in_φ':>8} {'in_E':>8}  |  {'out_x':>8} {'out_y':>8} {'out_θ':>8} {'out_φ':>8} {'out_E':>8}")
        print("-" * 110)
        for row in transmitted[:10]:
            print(f"{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.3f} {row[3]:8.3f} {row[4]:8.1f}  |  {row[5]:8.2f} {row[6]:8.2f} {row[7]:8.3f} {row[8]:8.3f} {row[9]:8.1f}")


if __name__ == "__main__":
    main()
