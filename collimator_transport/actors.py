"""
Actors for data acquisition at the collimator.

We use PhaseSpaceActor on two thin planes:
- Front plane: records incoming photons just before entering the collimator.
- Back plane: records outgoing photons that escaped through the backplane.
"""
import opengate as gate
from pathlib import Path


def add_collimator_phase_space_actors(sim, col_front_plane, col_back_plane, prefix="collimator"):
    """
    Add PhaseSpaceActor instances for incoming and outgoing photons.

    Incoming:
      - Attached to front plane.
      - Records position (x, y), direction, energy, track/event IDs.

    Outgoing:
      - Attached to back plane.
      - Same attributes.
      - By geometry, only photons exiting through the backplane are scored.
        Photons absorbed or leaving elsewhere simply do not appear and can be
        treated as having outgoing = 0 in post-processing.
    """
    subdir = Path("phsp")

    # Incoming phase space (front plane)
    ps_in = sim.add_actor("PhaseSpaceActor", name="ps_incoming")
    ps_in.attached_to = col_front_plane.name
    ps_in.attributes = [
        "KineticEnergy",
        "Weight",
        "PrePosition",
        "PreDirection",
        "ParticleName",
        "GlobalTime",
        "EventID",
        "TrackID",
        "ParentID",
    ]
    ps_in.output_filename = subdir / f"{prefix}_incoming.root"
    ps_in.steps_to_store = "entering"

    # Outgoing phase space (back plane)
    ps_out = sim.add_actor("PhaseSpaceActor", name="ps_outgoing")
    ps_out.attached_to = col_back_plane.name
    ps_out.attributes = [
        "KineticEnergy",
        "Weight",
        "PostPosition",
        "PostDirection",
        "ParticleName",
        "GlobalTime",
        "EventID",
        "TrackID",
        "ParentID",
    ]
    ps_out.output_filename = subdir / f"{prefix}_outgoing.root"
    ps_out.steps_to_store = "exiting"

    # Only gammas
    f = sim.add_filter("ParticleFilter", "phsp_gamma_filter")
    f.particle = "gamma"
    ps_in.filters.append(f)
    ps_out.filters.append(f)

    # Return actors in case caller wants to tweak further
    return ps_in, ps_out

