"""
Physics setup: enable lead characteristic X-rays (~80 keV) by
enabling fluorescence and setting production cuts low in the collimator.
"""
import opengate as gate


def configure_physics(sim, collimator, hole_region):
    """
    Configure physics so that characteristic X-rays of lead (~80 keV) can be
    generated in the collimator.

    - Enable fluorescence (fluo) in EM parameters.
    - Set production cuts in the collimator (and hole region) low enough
      that low-energy secondaries (e.g. leading to Pb K X-rays) are produced.
    """
    mm = gate.g4_units.mm

    # Enable fluorescence so Geant4 can produce characteristic X-rays (e.g. Pb K ~72–88 keV)
    sim.physics_manager.em_parameters.fluo = True

    # Production cut = range threshold: secondaries are produced only if their
    # range in the material is above this. Use a small cut (0.1 mm) in lead
    # so that low-energy secondaries and ~80 keV X-rays are generated.
    cut_lead_mm = 0.1

    for vol in (collimator, hole_region):
        sim.physics_manager.set_production_cut(vol.name, "gamma", cut_lead_mm * mm)
        sim.physics_manager.set_production_cut(vol.name, "electron", cut_lead_mm * mm)
        sim.physics_manager.set_production_cut(vol.name, "positron", cut_lead_mm * mm)
