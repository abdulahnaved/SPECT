"""
Flood source: plane 550×405 mm in front of the collimator.
Gamma, uniform 20–250 keV, hemisphere toward the collimator (+Z).
"""
import opengate as gate


def add_flood_source(sim, source_plane_z_mm=-20.0, n_primaries=1_000_000):
    """
    Add a flood (plane) source in front of the collimator.

    - Plane: 550 × 405 mm (X, Y), thin in Z, centered at (0, 0, source_plane_z_mm).
    - Particle: gamma.
    - Energy: uniform 20–250 keV.
    - Direction: hemisphere toward the collimator (toward +Z).

    In Geant4 spherical coords, direction z = -cos(θ). So θ in [90°, 180°]
    gives the +Z hemisphere.
    """
    mm = gate.g4_units.mm
    keV = gate.g4_units.keV
    deg = gate.g4_units.deg

    src = sim.add_source("GenericSource", "flood_source")
    src.particle = "gamma"

    src.n = n_primaries

    # Position: thin box = plane, 550 × 405 mm, centered at given Z
    src.position.type = "box"
    src.position.size = [550.0 * mm, 405.0 * mm, 0.1 * mm]
    src.position.translation = [0.0, 0.0, source_plane_z_mm * mm]

    # Direction: hemisphere toward +Z (toward the collimator)
    # Geant4: z = -cos(θ). θ ∈ [90°, 180°] → +Z hemisphere
    src.direction.type = "iso"
    src.direction.theta = [90 * deg, 180 * deg]
    src.direction.phi = [0 * deg, 360 * deg]

    # Energy: uniform 20–250 keV
    src.energy.type = "range"
    src.energy.min_energy = 20 * keV
    src.energy.max_energy = 250 * keV

    return src
