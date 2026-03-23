"""
Build and optionally run a single SPECT collimator simulation batch.

Each batch gets its own:
  - Random seed (for reproducibility)
  - Output directory (so ROOT files don't collide between parallel workers)
"""
import opengate as gate
from pathlib import Path

from .geometry import build_world_and_collimator
from .source import add_flood_source
from .physics import configure_physics
from .actors import add_collimator_phase_space_actors


def build_simulation(
    n_primaries: int = 1_000_000,
    seed: int = 42,
    batch_id: int = 0,
    output_dir: str = "output",
):
    """
    Construct a fully configured simulation for one batch.

    Args:
        n_primaries: number of gamma primaries to shoot.
        seed: fixed random seed for reproducibility.
        batch_id: integer label; output files go into output_dir/batch_<id>/.
        output_dir: top-level output folder.
    """
    sim = gate.Simulation()

    # Reproducibility
    sim.random_seed = seed
    sim.random_engine = "MersenneTwister"

    # Per-batch output directory
    batch_dir = Path(output_dir) / f"batch_{batch_id:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    sim.output_dir = str(batch_dir)

    # Geometry
    world, collimator, hole_region, col_front_plane, col_back_plane = (
        build_world_and_collimator(sim)
    )

    # Source
    add_flood_source(sim, source_plane_z_mm=-20.0, n_primaries=n_primaries)

    # Physics
    configure_physics(sim, collimator, hole_region)

    # Actors (phase-space at front/back planes)
    add_collimator_phase_space_actors(sim, col_front_plane, col_back_plane)

    return sim


def run_batch(
    n_primaries: int = 1_000_000,
    seed: int = 42,
    batch_id: int = 0,
    output_dir: str = "output",
):
    """Build and run one simulation batch. Returns the batch output directory."""
    sim = build_simulation(
        n_primaries=n_primaries,
        seed=seed,
        batch_id=batch_id,
        output_dir=output_dir,
    )
    sim.run()
    batch_dir = Path(output_dir) / f"batch_{batch_id:04d}"
    print(f"[batch {batch_id}] done → {batch_dir}")
    return str(batch_dir)
