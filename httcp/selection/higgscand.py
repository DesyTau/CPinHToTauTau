# coding: utf-8

"""
Prepare h-Candidate from SelectionResult: selected lepton indices & channel_id [trigger matched]
"""
import functools
from typing import Optional
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.util import DotDict

from httcp.util import enforce_hcand_type, enforce_tauprods_type, trigger_object_matching, hlt_path_fired, has_pt_greater_equal, has_delta_r_less_equal, HLT_path_matching

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@selector(
    uses={
        "Tau.decayModePNet",
    } | {f"{part}.IP{par}"
         for part in ["Electron", "Muon", "Tau"]
         for par in ["x", "y", "z"]} | {f"{part}.ip_sig" for part in ["Electron", "Muon", "Tau"]},
    produces={
        "hcand.pt", "hcand.eta", "hcand.phi", "hcand.mass", "hcand.charge", "hcand.rawIdx", "hcand.decayMode",  "hcand.decayModePNet",
        "hcand.IPx", "hcand.IPy", "hcand.IPz", "hcand.ip_sig"
    },
    exposed=False,
)
def higgscand(
        self: Selector,
        events: ak.Array,
        trigger_results: SelectionResult,
        hcand_pair: ak.Array,
        domatch: Optional[bool] = False,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:

    # Extraction of the indices from Hcand_pair
    etau_objects_idx = hcand_pair[:, 0].rawIdx
    etau_e_idx = etau_objects_idx[:, 0:1]
    etau_tau_idx = etau_objects_idx[:, 1:2]

    mutau_objects = hcand_pair[:, 1].rawIdx
    mutau_mu_idx = mutau_objects[:, 0:1]
    mutau_tau_idx = mutau_objects[:, 1:2]

    tautau_objects = hcand_pair[:, 2].rawIdx
    tautau_tau1_idx = tautau_objects[:, 0:1]
    tautau_tau2_idx = tautau_objects[:, 1:2]

    # Objects to match
    e_to_match = events.Electron[etau_e_idx]
    tau_etau_to_match = events.Tau[etau_tau_idx]

    mu_to_match = events.Muon[mutau_mu_idx]
    tau_mutau_to_match = events.Tau[mutau_tau_idx]

    tau1_tautau_to_match = events.Tau[tautau_tau1_idx]
    tau2_tautau_to_match = events.Tau[tautau_tau2_idx]

    ### Perfoming the trigger object matching is required###
    if domatch:
        hcand_array, etau_channel_mask, mutau_channel_mask, tautau_channel_mask = HLT_path_matching(events,
                                                                                                    trigger_results,
                                                                                                    mu_to_match, e_to_match,
                                                                                                    tau_mutau_to_match,
                                                                                                    tau_etau_to_match,
                                                                                                    tau1_tautau_to_match,
                                                                                                    tau2_tautau_to_match,
                                                                                                    hcand_pair)
    else:
        hcand_array = hcand_pair
    var_dict = {"pt"            : "float64",
            "eta"           : "float64",
            "phi"           : "float64",
            "mass"          : "float64",
            "charge"        : "int64",
            "decayMode"     : "int64",
            "decayModePNet" : "int64",
            "rawIdx"        : "int64",
            "IPx"           : "float64",
            "IPy"           : "float64",
            "IPz"           : "float64",
            "ip_sig"        : "float64",
            }
    hcand_array_type_fix = enforce_hcand_type(hcand_array, var_dict)

    # Extraction of the indices from Hcand_pair after matching

    etau_objects_idx = hcand_array_type_fix[:, 0].rawIdx
    etau_e_idx = etau_objects_idx[:, 0:1]
    etau_tau_idx = etau_objects_idx[:, 1:2]

    mutau_objects = hcand_array_type_fix[:, 1].rawIdx
    mutau_mu_idx = mutau_objects[:, 0:1]
    mutau_tau_idx = mutau_objects[:, 1:2]

    tautau_objects = hcand_array_type_fix[:, 2].rawIdx
    tautau_tau1_idx = tautau_objects[:, 0:1]
    tautau_tau2_idx = tautau_objects[:, 1:2]

    # Ensure that we save all the taus and we remove the duplicates
    tau_indices = ak.concatenate(
        [etau_tau_idx, mutau_tau_idx, tautau_tau1_idx, tautau_tau2_idx], axis=1)
    max_length = ak.max(ak.num(tau_indices, axis=1))
    tau_indices = ak.pad_none(tau_indices, target=max_length)
    tau_indices = ak.fill_none(tau_indices, -1)[:, :, None]
    tau_indices = ak.flatten(tau_indices, axis=2)
    tau_indices = tau_indices[tau_indices >= 0]

    hcand_electron_indices = ak.values_astype(etau_e_idx, np.int64)
    hcand_muon_indices = ak.values_astype(mutau_mu_idx, np.int64)
    hcand_tau_indices = ak.values_astype(tau_indices, np.int64)

    sel_hcand = ak.sum(ak.num(hcand_array_type_fix, axis=2), axis=1) == 2

    empty_hcand_array = hcand_array_type_fix[:, :0]

    hcand_array_skim = ak.where(
        sel_hcand, hcand_array_type_fix, empty_hcand_array)

    hcand_array_One_Higgs_cand = ak.flatten(hcand_array_skim, axis=2)
    #if Impact parameter is nan, these events should be skimmed
    if ak.any(np.isnan(hcand_array_One_Higgs_cand.IPx)) :
        hcand_array_One_Higgs_cand = ak.where(ak.any(np.isnan(hcand_array_One_Higgs_cand.IPx),axis = 1),hcand_array_One_Higgs_cand, hcand_array_One_Higgs_cand[..., :0])
    events = set_ak_column(events, "hcand", hcand_array_One_Higgs_cand)

    return events, hcand_muon_indices, hcand_electron_indices, hcand_tau_indices, etau_channel_mask, mutau_channel_mask, tautau_channel_mask, hcand_array_One_Higgs_cand, SelectionResult(
        steps={
            "single_hcand": sel_hcand,
        },
    )


def select_tauprods(hcand_idx, tauprods):

    field_type_dict = {
        'eta': 'float64',
        'pdgId': 'int64',
        'phi': 'float64',
        'pt': 'float64',
        'tauIdx': 'int64',
        'mass': 'float64',
        'charge': 'int64'
    }

    tauprods = enforce_tauprods_type(tauprods, field_type_dict)
    tauprods_idx = tauprods.tauIdx

    max_length = ak.max(ak.num(tauprods_idx, axis=1))
    tauprods_idx = ak.pad_none(tauprods_idx, target=max_length)
    hcand_idx = ak.pad_none(hcand_idx, target=max_length)
    hcandprod_mask = hcand_idx == tauprods_idx
    hcandprod_mask = ak.fill_none(hcandprod_mask, -1)

    hcandprods = tauprods[hcandprod_mask >= 0]

    return hcandprods


def is_pion(prods): return ((np.abs(prods.pdgId) == 211))
#                            | (np.abs(prods.pdgId) == 321))

def is_photon(prods): return prods.pdgId == 22

def has_one_pion(prods): return (ak.sum(is_pion(prods),   axis=1) == 1)

def has_three_pions(prods): return (ak.sum(is_pion(prods),   axis=1) == 3)

def has_photons(prods): return (ak.sum(is_photon(prods), axis=1) > 0)

def has_no_photons(prods): return (ak.sum(is_photon(prods), axis=1) == 0)


@selector(
    uses={
        "TauProd.pdgId",
    },
    produces={
        "TauProd.mass", "TauProd.charge",
    },
    exposed=False,
)
def assign_tauprod_mass_charge(
        self: Selector,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    # pionp = 211
    # pionm = -211
    # kaonp = 321
    # kaonm = -321
    # gamma = 22

    # mass = ak.where(np.abs(events.TauProd.pdgId) == pionp,
    #                 0.13957,
    #                 ak.where(np.abs(events.TauProd.pdgId) == kaonp,
    #                          0.493677,
    #                          ak.where(events.TauProd.pdgId == gamma,
    #                                   0.0, 0.0))
    #                 )
    # charge = ak.where(((events.TauProd.pdgId == pionp) | (events.TauProd.pdgId == kaonp)),
    #                   1.0,
    #                   ak.where(((events.TauProd.pdgId == pionm) | (events.TauProd.pdgId == kaonm)),
    #                            -1.0,
    #                            0.0)
    #                   )

    # events = set_ak_column(events, "TauProd.mass", mass)
    # events = set_ak_column(events, "TauProd.charge", charge)
    
    #https://pdg.lbl.gov/2023/listings/particle_properties.html
    
    part_dict=DotDict.wrap({
        "pion_pm"   : {'mass'    : 0.13957039, #GeV
                       'pdg_id'  : 211},
        "pion_0"    : {'mass'    : 0.1349768, #GeV
                       'pdg_id'  : 111},
        "gamma"     : {'mass'    : 0.0, #GeV
                       'pdg_id'  : 22},
        "kaon_pm"   : {'mass'    : 0.493677, #GeV
                       'pdg_id'  : 321},
        "ele_pm"    : {'mass'    : 0.00051099895, #GeV
                       'pdg_id'  : 11},
        "muon_pm"   : {'mass'    : 0.1056583755, #GeV
                       'pdg_id'  : 13},
        })
    mass = ak.zeros_like(ak.local_index(events.TauProd.pdgId), dtype=np.float32)
    charge = ak.zeros_like(ak.local_index(events.TauProd.pdgId), dtype=np.int32)
    for part in part_dict:
       
        prod_id = events.TauProd.pdgId
        mass = ak.where(np.abs(prod_id)==part_dict[part].pdg_id,
                        part_dict[part].mass,
                        mass)
        if '_pm' in part:
            charge = ak.where(np.abs(prod_id)==part_dict[part].pdg_id,
                              np.sign(prod_id),
                              charge)
    
    events = set_ak_column_f32(events, "TauProd.mass", mass)
    events = set_ak_column_i32(events, "TauProd.charge", charge)
    return events


@selector(
    uses={
        "channel_id", "TauProd.*", assign_tauprod_mass_charge,
    },
    produces={
        "hcandprod.pt", "hcandprod.eta", "hcandprod.phi", "hcandprod.mass", "hcandprod.charge", "hcandprod.pdgId","hcandprod.tauIdx",
        assign_tauprod_mass_charge,
    },
    exposed=False,
)
def higgscandprod(
        self: Selector,
        events: ak.Array,
        hcand_array: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    # etau_id   = self.config_inst.get_channel("etau").id
    # mutau_id  = self.config_inst.get_channel("mutau").id
    # tautau_id = self.config_inst.get_channel("tautau").id

    # events   = self[assign_tauprod_mass_charge](events)

    # tauprods = events.TauProd

    # hcand1 = events.hcand[:, 0:1]
    # hcand2 = events.hcand[:, 1:2]

    # hcand1_idx = hcand1.rawIdx
    # hcand2_idx = hcand2.rawIdx

    # hcand1prods = ak.where(events.channel_id == tautau_id,
    #                        select_tauprods(hcand1_idx, tauprods),
    #                        tauprods[:, :0])
    # hcand2prods = ak.where(((events.channel_id == etau_id) | (events.channel_id == mutau_id) | (events.channel_id == tautau_id)),
    #                        select_tauprods(hcand2_idx, tauprods),
    #                        tauprods[:, :0])

    # dummy = (events.event >= 0)[:, None]
    # hcand1_mask = ak.where(hcand1.decayMode == 0,
    #                        (has_one_pion(hcand1prods) &
    #                         has_no_photons(hcand1prods)),
    #                        ak.where(((hcand1.decayMode == 1) | (hcand1.decayMode == 2)),
    #                                 (has_one_pion(hcand1prods) &
    #                                  has_photons(hcand1prods)),
    #                                 ak.where(hcand1.decayMode == 10,
    #                                          has_three_pions(hcand1prods),
    #                                          ak.where(hcand1.decayMode == 11,
    #                                                   (has_three_pions(hcand1prods)
    #                                                    & has_photons(hcand1prods)),
    #                                                   dummy)
    #                                          )
    #                                 )
    #                        )
    # hcand2_mask = ak.where(hcand2.decayMode == 0,
    #                        (has_one_pion(hcand2prods) &
    #                         has_no_photons(hcand2prods)),
    #                        ak.where(((hcand2.decayMode == 1) | (hcand2.decayMode == 2)),
    #                                 (has_one_pion(hcand2prods) &
    #                                  has_photons(hcand2prods)),
    #                                 ak.where(hcand2.decayMode == 10,
    #                                          has_three_pions(hcand2prods),
    #                                          ak.where(hcand2.decayMode == 11,
    #                                                   (has_three_pions(hcand2prods)
    #                                                    & has_photons(hcand2prods)),
    #                                                   dummy)
    #                                          )
    #                                 )
    #                        )

    # hcand_prod_mask = ak.concatenate([hcand1_mask, hcand2_mask], axis=1)

    # hcand_prods = ak.concatenate(
    #     [hcand1prods[:, None], hcand2prods[:, None]], axis=1)

    # hcand_prods_array = enforce_hcand_type(ak.from_regular(hcand_prods),
    #                                        {"pt": "float64",
    #                                         "eta": "float64",
    #                                         "phi": "float64",
    #                                         "mass": "float64",
    #                                         "charge": "int64",
    #                                         "pdgId": "int64",
    #                                         "tauIdx": "int64"}
    #                                        )
    # hcand_prods_array = ak.from_regular(hcand_prods)
    # events = set_ak_column(events, "hcandprod", hcand_prods_array)

    events   = self[assign_tauprod_mass_charge](events)

    tauprods = events.TauProd
    tau_mask = events.hcand.decayMode >=0
    tau_idx = ak.mask(events.hcand.rawIdx,tau_mask)
    
    tauprods_mask = tauprods.tauIdx
    idx_pairs = ak.cartesian([tau_idx,tauprods.tauIdx], axis=1, nested=True)
    tau_idx_b, tau_prod_idx_b = ak.unzip(idx_pairs)
    prod_mask = ak.fill_none(tau_idx_b == tau_prod_idx_b, False)
    evt_masks = []
    evt_dm_only_masks = []
    tau_prods = []
    for idx in range(2): #iteate over taus
        tau = ak.firsts(events.hcand[:,idx:idx+1])
        tau_prod = tauprods[ak.flatten(prod_mask[:,idx:idx+1],axis=2)]
        false_prod_mask = ak.zeros_like(ak.local_index(tau_prod,axis=1), dtype=np.bool_)
        mask = ak.ones_like(ak.local_index(tau_prod,axis=1), dtype=np.bool_)
        #DM0
        mask = ak.fill_none(tau.decayMode==0, False) & has_one_pion(tau_prod) #& has_no_photons(tau_prod)
        #DM1
        mask = mask | ak.fill_none(tau.decayMode==1, False) & has_one_pion(tau_prod) & has_photons(tau_prod)
        #DM10
        mask = mask | ak.fill_none(tau.decayMode==10, False) & has_three_pions(tau_prod) #& has_no_photons(tau_prod)
        #DM11
        mask = mask | ak.fill_none(tau.decayMode==11, False) & has_three_pions(tau_prod) & has_no_photons(tau_prod)
        #DM -1, -2 for electron / muon
        mask = mask | ak.fill_none(tau.decayMode < 0, False)
        evt_masks.append(mask)
        
        dm_only_mask = (ak.fill_none(tau.decayMode==0, False) | ak.fill_none(tau.decayMode==1, False) |
        ak.fill_none(tau.decayMode==10, False) | ak.fill_none(tau.decayMode==11, False) | ak.fill_none(tau.decayMode < 0, False))
        
        evt_dm_only_masks.append(dm_only_mask)
        tau_prods.append(tau_prod[:,None])
                
    etau_id   = self.config_inst.get_channel("etau").id
    mutau_id  = self.config_inst.get_channel("mutau").id
    tautau_id = self.config_inst.get_channel("tautau").id

    joint_evt_mask = (events.channel_id == tautau_id) & evt_masks[0] & evt_masks[1]
    joint_evt_mask = joint_evt_mask + (((events.channel_id == etau_id) | (events.channel_id == mutau_id)) & evt_masks[0]) 
    
    joint_evt_dm_mask = (events.channel_id == tautau_id) & evt_dm_only_masks[0] & evt_dm_only_masks[1]
    joint_evt_dm_mask = joint_evt_mask + (((events.channel_id == etau_id) | (events.channel_id == mutau_id)) & evt_dm_only_masks[0]) 
        
    hcand_prods = ak.concatenate(tau_prods, axis=1)

    hcand_prods_array = enforce_hcand_type(ak.from_regular(hcand_prods),
                                           {"pt": "float64",
                                            "eta": "float64",
                                            "phi": "float64",
                                            "mass": "float64",
                                            "charge": "int64",
                                            "pdgId": "int64",
                                            "tauIdx": "int64"}
                                           )
    hcand_prods_array = ak.from_regular(hcand_prods)
    events = set_ak_column(events, "hcandprod", hcand_prods_array)

    return events, SelectionResult(
        steps={
            "decay_mode_mask"   : joint_evt_dm_mask,
            "decay_prods_are_ok": joint_evt_mask,
        },
    )
