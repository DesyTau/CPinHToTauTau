"""
Calibration methods.
"""
import functools

from columnflow.calibration import Calibrator, calibrator
from MSSM_H_tt.calibration.jets import jec, jer
from MSSM_H_tt.calibration.type1_met_corr import jets_puppimet_only_ak4
from MSSM_H_tt.calibration.tau import tau_energy_scale
from MSSM_H_tt.calibration.electron import electron_smearing_scaling
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
import law

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@calibrator(
    uses={
       jec,
       jer,
       jets_puppimet_only_ak4,
       tau_energy_scale,
       electron_smearing_scaling,
       deterministic_seeds,
       "Jet.pt",
       "Jet.eta",
       "Jet.phi",
       "Jet.area",
       "Jet.rawFactor",
       "PuppiMET.pt",
       "PuppiMET.phi",
       "PuppiMET.ptUnclusteredUp",
       "PuppiMET.phiUnclusteredUp",
       "PuppiMET.ptUnclusteredDown",
       "PuppiMET.phiUnclusteredDown",
       "Electron.phi",
       "Tau.phi",
       "Tau.pt",
       "run",
       "luminosityBlock",
       "event",
    },
    produces={
        jec,
        jer,
        jets_puppimet_only_ak4,
        tau_energy_scale,
        electron_smearing_scaling,
        deterministic_seeds,
    },
)
def main(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:

    events = self[deterministic_seeds](events, **kwargs)

    print("Performing electron scaling and smearing correction...")
    events = self[electron_smearing_scaling](events, **kwargs)
    print("Electron scaling and smearing correction... SUCCEEDED")

    if self.dataset_inst.is_mc and (self.config_inst.channels.names()[0] != "emu"):
        print("Performing tau energy scale correction...")
        events = self[tau_energy_scale](events, **kwargs)
    
    puppimet_pt_rel_unc_up = (events.PuppiMET.ptUnclusteredUp/events.PuppiMET.pt)
    puppimet_pt_rel_unc_down = (events.PuppiMET.ptUnclusteredDown/events.PuppiMET.pt)
    puppimet_phi_rel_unc_up = (events.PuppiMET.phiUnclusteredUp/events.PuppiMET.phi)
    puppimet_phi_rel_unc_down = (events.PuppiMET.phiUnclusteredDown/events.PuppiMET.phi)
    print("Performing JEC...")
    events = self[jec](events, **kwargs)
    if self.dataset_inst.is_mc:
        print("Performing JER...")
        events = self[jer](events, **kwargs)
    print("Propagating JEC+JER to PuppiMET once, without modifying Jet branches...")
    events = self[jets_puppimet_only_ak4](events, **kwargs)
    events = set_ak_column_f32(events, "PuppiMET.ptUnclusteredUp", ak.nan_to_num(events.PuppiMET.pt*puppimet_pt_rel_unc_up, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT ))
    events = set_ak_column_f32(events, "PuppiMET.phiUnclusteredUp", ak.nan_to_num(events.PuppiMET.phi*puppimet_phi_rel_unc_up, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT))
    events = set_ak_column_f32(events, "PuppiMET.ptUnclusteredDown", ak.nan_to_num(events.PuppiMET.pt*puppimet_pt_rel_unc_down, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT))
    events = set_ak_column_f32(events, "PuppiMET.phiUnclusteredDown", ak.nan_to_num(events.PuppiMET.phi*puppimet_phi_rel_unc_down, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT))

    return events