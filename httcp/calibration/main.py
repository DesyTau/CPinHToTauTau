"""
Calibration methods.
"""
import functools

from columnflow.calibration import Calibrator, calibrator
from httcp.calibration.jets import jets, jec, jer
from httcp.calibration.met import met_phi
from httcp.calibration.tau import tau_energy_scale
from httcp.calibration.electron import electron_scaling, electron_smearing, electron_smearing_scaling
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@calibrator(
    uses={
        jec, tau_energy_scale, electron_smearing_scaling, deterministic_seeds, 
    },
    produces={
        jec, tau_energy_scale, electron_smearing_scaling, deterministic_seeds, "Jet.pt_no_jec", "Jet.eta_no_jec",
        "Jet.phi_no_jec", "Jet.mass_no_jec", "PuppiMET.pt_no_jec", "PuppiMET.phi_no_jec", "nJet", "Electron.pt_no_scaling_smearing",
    },
)
def main(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:

    events = self[deterministic_seeds](events, **kwargs)
    
    #Jets variables before applying energy corrections
    events = set_ak_column_f32(events, "Jet.pt_no_jec", events.Jet.pt)
    events = set_ak_column_f32(events, "Jet.phi_no_jec", events.Jet.phi)
    events = set_ak_column_f32(events, "Jet.eta_no_jec", events.Jet.eta)
    events = set_ak_column_f32(events, "Jet.mass_no_jec", events.Jet.mass)
    events = set_ak_column_f32(events, "nJet", events.nJet)
    
    #PuppiMET variables before applying energy corrections
    events = set_ak_column_f32(events, "PuppiMET.pt_no_jec", events.PuppiMET.pt)
    events = set_ak_column_f32(events, "PuppiMET.phi_no_jec", events.PuppiMET.phi)

    #events = self[jets](events, **kwargs)
    
    events = self[jec](events, **kwargs)

    #events = self[met_phi](events, **kwargs)
    #events = self[jer](events, **kwargs)
    events = set_ak_column_f32(events, "Electron.pt_no_scaling_smearing", events.Electron.pt)
    
    print("Performing electron scaling and smearing correction...")
    events = self[electron_smearing_scaling](events, **kwargs) 
    
    if self.dataset_inst.is_mc: 
    #Apply tau energy scale correction
        print("Performing tau energy scale correction...")
        events = self[tau_energy_scale](events, **kwargs)

    return events