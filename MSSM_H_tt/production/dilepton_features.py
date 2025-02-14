import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT, Route, optional_column as optional
from columnflow.production.util import attach_coffea_behavior
from MSSM_H_tt.util import get_lep_p4
ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

def hcand_mt(lep: ak.Array, MET: ak.Array) -> ak.Array:
    print("Producing mT...")
    delta_phi = lep.phi - MET.phi
    delta_phi = ak.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
    delta_phi = ak.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
    #cos_dphi = np.cos(lep.delta_phi(MET))
    cos_dphi = np.cos(delta_phi)
    mT_values = np.sqrt(2*lep.pt*MET.pt * (1 - cos_dphi))
    return ak.fill_none(mT_values, EMPTY_FLOAT)

@producer(
    uses={
        'hcand_*', 'PuppiMET*'
    },
    produces={
        'hcand_*'
    },
)
def hcand_fields(
        self: Producer, 
        events: ak.Array,
        **kwargs
) -> ak.Array:
    channels = self.config_inst.channels.names()
    ch_objects = self.config_inst.x.ch_objects
    for ch_str in channels:
        hcand = events[f'hcand_{ch_str}']
        p4 = {}
        for the_lep in hcand.fields: p4[the_lep] = get_lep_p4(hcand[the_lep]) 
        
        pair = p4['lep0'] + p4['lep1']
        dilep_mass = pair.mass
        hcand['mass'] = ak.where(dilep_mass > 0, dilep_mass , EMPTY_FLOAT)
        dilep_pt = pair.pt
        hcand['pt'] = ak.where(dilep_pt > 0, dilep_pt , EMPTY_FLOAT)
        delta_r = ak.flatten(p4['lep0'].metric_table(p4['lep1']), axis=2)
        hcand['delta_r'] = ak.where(delta_r > 0, delta_r , EMPTY_FLOAT)
        hcand['rel_charge'] = hcand.lep0.charge * hcand.lep1.charge
        if ch_str !=' tautau':
            mt = hcand_mt(p4['lep0'], events.PuppiMET)
            hcand['mt'] = ak.where(mt > 0, mt , EMPTY_FLOAT)
            
        
        events = set_ak_column(events, f'hcand_{ch_str}', hcand) 
    return events

    
   
