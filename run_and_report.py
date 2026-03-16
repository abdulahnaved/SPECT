"""
Run the SPECT collimator simulation and print basic phase-space stats.

Usage (from project root, with opengate_env activated):

    python run_and_report.py
"""

import sys

import uproot

from collimator_transport.main import build_simulation


def run_simulation():
    sim = build_simulation()
    sim.run()


def summarize_phsp():
    try:
        tin = uproot.open("phsp/collimator_incoming.root")["ps_incoming"]
        tout = uproot.open("phsp/collimator_outgoing.root")["ps_outgoing"]
    except Exception as e:
        print(f"Could not open phase-space ROOT files: {e}")
        return

    # Energies in MeV
    E_in = tin["KineticEnergy"].array()
    E_out = tout["KineticEnergy"].array()

    n_in = len(E_in)
    n_out = len(E_out)

    print("----- Phase-space summary -----")
    print(f"N incoming photons: {n_in}")
    print(f"N outgoing photons: {n_out}")
    if n_in > 0:
        print(f"Transmission fraction (N_out / N_in): {n_out / n_in:.6g}")

    if n_in > 0:
        print(
            "First 5 incoming energies (keV):",
            (E_in[:5] * 1000).tolist(),
        )
    if n_out > 0:
        print(
            "First 5 outgoing energies (keV):",
            (E_out[:5] * 1000).tolist(),
        )


def main():
    run_simulation()
    summarize_phsp()


if __name__ == "__main__":
    sys.exit(main())

