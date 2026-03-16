import opengate as gate
import math


def build_world_and_collimator(sim):
    """
    Configure the world and add the collimator volume.

    World:
      - Filled with air
      - Same size as the collimator in X and Y
      - Extra thickness in Z to host a plane source in front of the collimator

    Collimator:
      - 550 x 405 x 23.8 mm (X, Y, Z)
      - Lead material (G4_Pb)
    """

    # Collimator outer dimensions (mm)
    col_x = 550.0
    col_y = 405.0
    col_z = 23.8

    # Hole-region dimensions (mm) (central part of the collimator)
    hole_x = 545.0
    hole_y = 400.0
    hole_z = col_z

    # Hexagonal holes parameters (mm)
    hole_diameter = 1.2
    septum_thickness = 0.2

    # World: same X/Y, extra space in Z for a plane source
    # Here we add 40 mm in front of the collimator for the source plane
    extra_z_front = 40.0
    extra_z_back = 10.0

    world = sim.world
    world.size = [col_x, col_y, col_z + extra_z_front + extra_z_back]
    world.material = "G4_AIR"

    # Place the collimator so that there is space on the -Z side for a plane source
    collimator = sim.add_volume("Box", "collimator")
    collimator.mother = world.name
    collimator.size = [col_x, col_y, col_z]

    # Shift collimator towards +Z so that the free space is mainly on the -Z side
    # World center is (0,0,0). We place the collimator so that its front face
    # (towards -Z) sits extra_z_front mm away from the world boundary.
    world_z_half = 0.5 * world.size[2]
    col_z_half = 0.5 * col_z
    # Position of collimator center along Z
    col_z_center = -world_z_half + extra_z_front + col_z_half
    collimator.translation = [0.0, 0.0, col_z_center]

    # Use Geant4 lead material; density is ~11.34 g/cm3, close to the requested 11.4
    collimator.material = "G4_Pb"

    # ------------------------------------------------------------------
    # Collimator holes: define the central "hole region" only for now
    # ------------------------------------------------------------------
    hole_region = sim.add_volume("Box", "collimator_hole_region")
    hole_region.mother = collimator.name
    hole_region.size = [hole_x, hole_y, hole_z]
    hole_region.translation = [0.0, 0.0, 0.0]
    hole_region.material = "G4_Pb"

    # Fill the hole_region with cylindrical holes arranged on a hexagonal (triangular) lattice.
    _fill_cylindrical_hex_holes(
        sim,
        mother=hole_region,
        hole_diameter=hole_diameter,
        septum_thickness=septum_thickness,
        extent_x=hole_x,
        extent_y=hole_y,
        height=hole_z,
    )

    # ------------------------------------------------------------------
    # Scoring planes for phase-space (incoming / outgoing)
    # ------------------------------------------------------------------
    # Front face (toward source) and back face (toward detector) positions
    col_front_z = col_z_center - col_z_half
    col_back_z = col_z_center + col_z_half

    # Thin air slabs slightly outside the collimator faces (avoid overlaps)
    plane_thickness = 0.1  # mm
    gap = 0.05  # mm offset from the collimator surface

    col_front_plane = sim.add_volume("Box", "collimator_front_plane")
    col_front_plane.mother = world.name
    col_front_plane.size = [hole_x, hole_y, plane_thickness]
    col_front_plane.translation = [0.0, 0.0, col_front_z - (plane_thickness / 2.0 + gap)]
    col_front_plane.material = "G4_AIR"

    col_back_plane = sim.add_volume("Box", "collimator_back_plane")
    col_back_plane.mother = world.name
    col_back_plane.size = [hole_x, hole_y, plane_thickness]
    col_back_plane.translation = [0.0, 0.0, col_back_z + (plane_thickness / 2.0 + gap)]
    col_back_plane.material = "G4_AIR"

    return world, collimator, hole_region, col_front_plane, col_back_plane


def _fill_cylindrical_hex_holes(
    sim,
    mother,
    hole_diameter: float,
    septum_thickness: float,
    extent_x: float,
    extent_y: float,
    height: float,
):
    """
    Approximate hexagonal holes with cylinders (Tubs) on a triangular lattice.

    - Cylinder diameter = requested hole diameter.
    - Center-to-center pitch in X = hole_diameter + septum_thickness.
    - Row spacing (Y) = pitch_x * sqrt(3) / 2 (triangular packing).

    Cylindrical holes are G4_AIR inside a lead bulk mother volume.
    """

    pitch_x = hole_diameter + septum_thickness
    pitch_y = pitch_x * math.sqrt(3.0) / 2.0

    # Number of holes that can fit along each direction (rough estimate)
    nx = int(extent_x // pitch_x)
    ny = int(extent_y // pitch_y)

    # For performance while prototyping, cap the number of holes.
    # You can increase these caps once you're happy with the geometry.
    nx_cap = 120
    ny_cap = 120
    if nx > nx_cap:
        nx = nx_cap
    if ny > ny_cap:
        ny = ny_cap

    r = 0.5 * hole_diameter

    created = 0
    for j in range(ny):
        # Center rows around 0 in Y
        y = (j - (ny - 1) / 2.0) * pitch_y

        # Stagger every other row to create triangular (hex-like) packing
        x_offset = 0.0 if (j % 2 == 0) else 0.5 * pitch_x

        for i in range(nx):
            x = (i - (nx - 1) / 2.0) * pitch_x + x_offset

            # Stay within the rectangular hole-region boundaries with a small margin
            if abs(x) > 0.5 * extent_x - r or abs(y) > 0.5 * extent_y - r:
                continue

            name = f"collimator_hole_{i}_{j}"
            hole = sim.add_volume("Tubs", name)
            hole.mother = mother.name
            hole.material = "G4_AIR"

            # Tubs parameters: rmin, rmax, dz, sphi, dphi (angles in rad)
            hole.rmin = 0.0
            hole.rmax = r
            hole.dz = 0.5 * height
            hole.sphi = 0.0
            hole.dphi = 2.0 * math.pi

            # Local coordinates inside mother volume
            hole.translation = [x, y, 0.0]

            created += 1

    print(f"Created {created} cylindrical holes inside the collimator.")

