import opengate as gate

from .geometry import build_world_and_collimator
from .source import add_flood_source
from .physics import configure_physics
from .actors import add_collimator_phase_space_actors


def build_simulation():
    """
    OpenGATE simulation: world, collimator with holes, flood source, physics
    for Pb characteristic X-rays, and phase-space actors at the collimator
    front/back planes.
    """
    sim = gate.Simulation()
    world, collimator, hole_region, col_front_plane, col_back_plane = build_world_and_collimator(sim)
    add_flood_source(sim, source_plane_z_mm=-20.0)
    configure_physics(sim, collimator, hole_region)
    add_collimator_phase_space_actors(sim, col_front_plane, col_back_plane)

    print("World size (mm):", world.size)
    print("World material:", world.material)
    print("Collimator size (mm):", collimator.size)
    print("Collimator material:", collimator.material)
    print("Collimator position (mm):", collimator.translation)
    print("Hole region size (mm):", hole_region.size)
    print("Hole region material:", hole_region.material)
    print("Flood source: 550×405 mm plane, gamma 20–250 keV, hemisphere toward +Z")
    print("Phase-space: incoming at front plane, outgoing at back plane")

    return sim


if __name__ == "__main__":
    sim = build_simulation()
    # Simple run configuration: shoot the requested number of primaries.
    sim.run()

