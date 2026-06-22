# coding: utf-8

"""
Main categories file for the Higgs CP analysis
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

from types import FunctionType
from copy import copy

ak = maybe_import("awkward")
np = maybe_import("numpy")

#
# categorizer functions used by categories definitions
#

def copy_function(fn, name):
    return FunctionType(
    copy(fn.__code__),
    copy(fn.__globals__),
    name=name,
    argdefs=copy(fn.__defaults__),
    closure=copy(fn.__closure__)
)
    
@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1

#Four general categories: etau, mutau, emu and tautau
@categorizer(uses={'hcand_etau.*'})
def cat_etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.num(events.hcand_etau.lep0.pt > 0, axis =1) > 0
    return events, mask 

@categorizer(uses={'hcand_mutau.*'})
def cat_mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.num(events.hcand_mutau.lep0.pt > 0, axis =1) > 0
    return events, mask

@categorizer(uses={'hcand_emu.*'})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.num(events.hcand_emu.lep0.pt > 0, axis =1) > 0
    return events, mask 

@categorizer(uses={'event', 'hcand_emu.lep1.pfRelIso04_all'})
def lep_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channel = self.config_inst.channels.names()[0] #We are processing a single channel at once
    if channel == 'emu': 
        isolation = events.hcand_emu.lep1.pfRelIso04_all < 0.2
    else:
        raise NotImplementedError(
                f'Can not find an isolation criteria for {channel} channel!')
    mask = ak.fill_none(ak.firsts(isolation, axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def lep_inv_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channel = self.config_inst.channels.names()[0] #We are processing a single channel at once
    if channel == 'emu':
        isolation = events.hcand_emu.lep1.pfRelIso04_all >= 0.2
        upper_lim = events.hcand_emu.lep1.pfRelIso04_all < 0.5
    else:
        raise NotImplementedError(
                f'Can not find an isolation criteria for {channel} channel!')
    mask = ak.fill_none(ak.firsts(isolation, axis=1),False)
    mask = mask & ak.fill_none(ak.firsts(upper_lim, axis=1),False)
    return events, mask


@categorizer(uses={'hcand_tautau.*'})
def cat_tautau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.num(events.hcand_tautau.lep0.pt > 0, axis =1) > 0
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def os_charge(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].rel_charge < 0), axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def ss_charge(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].rel_charge > 0), axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def mt_inv_cut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        if ch_str != 'tautau':
            mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].mt > 50), axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def mt_cut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        if ch_str != 'tautau':
            mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].mt <= 50), axis=1),False)
    return events, mask

@categorizer(uses={"N_b_jets"})
def Zero_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets == 0 
    return events, mask

@categorizer(uses={"N_b_jets"})
def One_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets == 1 
    return events, mask

@categorizer(uses={"N_b_jets"})
def At_least_1_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets >= 1
    return events, mask

@categorizer(uses={"N_b_jets"})
def At_least_2_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets >= 2 
    return events, mask

@categorizer(uses={"OC_lepton_veto"})
def OC_lepton_veto(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.OC_lepton_veto
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def deep_tau_inv_wp(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    deep_tau_vs_e_jet_wps = self.config_inst.x.deep_tau.vs_e_jet_wps
    deep_tau_vs_mu_wps = self.config_inst.x.deep_tau.vs_mu_wps
    
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for channel in channels:
        tau = events[f'hcand_{channel}'].lep1 
        channel_mask = ak.ones_like(events[f'hcand_{channel}'].lep1.rawIdx)
        if channel == 'mutau':
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSjet < deep_tau_vs_e_jet_wps["Medium"]) #This cut is reversed
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSe   >= deep_tau_vs_e_jet_wps["VVLoose"])
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSmu  >= deep_tau_vs_mu_wps["Tight"])
        elif channel == 'etau':
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSjet < deep_tau_vs_e_jet_wps["Medium"]) #This cut is reversed
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSe   >= deep_tau_vs_e_jet_wps["Tight"])
            channel_mask = channel_mask & (tau.idDeepTau2018v2p5VSmu  >= deep_tau_vs_mu_wps["VLoose"])
        elif "tautau":
            tau0 = events[f'hcand_{channel}'].lep0
            for the_tau in [tau, tau0]:
                channel_mask = channel_mask & (the_tau.idDeepTau2018v2p5VSjet < deep_tau_vs_e_jet_wps["Medium"]) #This cut is reversed
                channel_mask = channel_mask & (the_tau.idDeepTau2018v2p5VSe   >= deep_tau_vs_e_jet_wps["VVLoose"])
                channel_mask = channel_mask & (the_tau.idDeepTau2018v2p5VSmu  >= deep_tau_vs_mu_wps["VLoose"])
        mask = mask | ak.fill_none(ak.firsts(channel_mask, axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def tau_endcap(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
            mask = mask | ak.fill_none(ak.firsts((np.abs(events[f'hcand_{ch_str}'].lep1.eta) > 1.2), axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def tau_barrel(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
            mask = mask | ak.fill_none(ak.firsts((np.abs(events[f'hcand_{ch_str}'].lep1.eta) <= 1.2), axis=1),False)
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_low(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -35) & (events.D_zeta < -10)
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_mid(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -10) 
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_high(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= 30)
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -80)
    return events, mask
  
@categorizer(uses={'event', 'hcand_*'})
def tau_no_fakes(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channel = self.config_inst.channels.names()[0] #We are processing a single channel at once
    if self.dataset_inst.is_mc:
        mask = ak.fill_none(ak.firsts(events[f'hcand_{channel}'].lep1.genPartFlav!=0, axis=1),False)
    else:
        mask = ak.ones_like(events.event, dtype=np.bool_)
    return events, mask

# Higgs BDT four-class score categories ---------------------------------------

def _bdt_cat_mass(
    self: Categorizer,
    events: ak.Array,
    cat_id: int,
    mass: int,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Per-mass BDT region selection for the current four-class BDT.

    The BDT-score producer should write:

        bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)

    with the convention:

        0 -> ggphi
        1 -> bbphi
        2 -> dy
        3 -> tt
    """
    field = f"bdt_cat_M{mass}"

    if field not in events.fields:
        raise RuntimeError(
            f"Missing BDT category field '{field}'. "
            "Check that the BDT-score producer was run and that it writes "
            f"bdt_cat_M{mass}."
        )

    mask = events[field] == cat_id

    return events, ak.fill_none(mask, False)


def _bdt_cat_ggphi_and_bbphi_mass(
    self: Categorizer,
    events: ak.Array,
    mass: int,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Per-mass combined signal-like BDT region.

    Definition:

        P_sig = P_ggphi + P_bbphi

    Select events where:

        P_sig >= P_DY
        P_sig >= P_TT

    This is not the same as selecting:

        bdt_cat_M{mass} == 0 or bdt_cat_M{mass} == 1

    because an event can have neither P_ggphi nor P_bbphi individually maximal,
    while their sum is still larger than both P_DY and P_TT.
    """
    ggphi_field = f"bdt_raw_score_ggphi_M{mass}"
    bbphi_field = f"bdt_raw_score_bbphi_M{mass}"
    dy_field = f"bdt_raw_score_dy_M{mass}"
    tt_field = f"bdt_raw_score_tt_M{mass}"

    required_fields = [
        ggphi_field,
        bbphi_field,
        dy_field,
        tt_field,
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in events.fields
    ]

    if missing_fields:
        raise RuntimeError(
            "Missing BDT raw-score fields needed for the combined "
            f"ggphi_and_bbphi category for mass {mass}: {missing_fields}. "
            "Check that the BDT-score producer writes the raw scores:"
            f" {required_fields}."
        )

    p_sig = events[ggphi_field] + events[bbphi_field]
    p_dy = events[dy_field]
    p_tt = events[tt_field]

    mask = (
        (p_sig >= p_dy)
        & (p_sig >= p_tt)
    )

    return events, ak.fill_none(mask, False)


from MSSM_H_tt.config.mass_points import read_bdt_masses

MASS_POINTS = read_bdt_masses()


BDT_REGION_SPECS = {
    "ggphi": {
        "kind": "single",
        "cat_id": 0,
        "description": "BDT region where P_ggphi is maximal",
    },
    "bbphi": {
        "kind": "single",
        "cat_id": 1,
        "description": "BDT region where P_bbphi is maximal",
    },
    "ggphi_and_bbphi": {
        "kind": "signal_sum",
        "description": "BDT region where P_ggphi + P_bbphi is maximal",
    },
    "dy": {
        "kind": "single",
        "cat_id": 2,
        "description": "BDT region where P_DY is maximal",
    },
    "tt": {
        "kind": "single",
        "cat_id": 3,
        "description": "BDT region where P_TT is maximal",
    },
}


BDT_REGION_DESCRIPTIONS = {
    region_name: spec["description"]
    for region_name, spec in BDT_REGION_SPECS.items()
}


for mass in MASS_POINTS:
    for region_name, spec in BDT_REGION_SPECS.items():
        if spec["kind"] == "single":
            cat_id = spec["cat_id"]

            # Capture loop variables via defaults to avoid late binding.
            tmp_func = (
                lambda self, events, _cat_id=cat_id, _mass=mass, **kwargs:
                    _bdt_cat_mass(
                        self,
                        events,
                        cat_id=_cat_id,
                        mass=_mass,
                        **kwargs,
                    )
            )

            uses = {
                f"bdt_cat_M{mass}",
            }

        elif spec["kind"] == "signal_sum":
            # Capture loop variable via default to avoid late binding.
            tmp_func = (
                lambda self, events, _mass=mass, **kwargs:
                    _bdt_cat_ggphi_and_bbphi_mass(
                        self,
                        events,
                        mass=_mass,
                        **kwargs,
                    )
            )

            uses = {
                f"bdt_raw_score_ggphi_M{mass}",
                f"bdt_raw_score_bbphi_M{mass}",
                f"bdt_raw_score_dy_M{mass}",
                f"bdt_raw_score_tt_M{mass}",
            }

        else:
            raise ValueError(
                f"Unknown BDT region kind '{spec['kind']}' "
                f"for region '{region_name}'."
            )

        globals()[f"bdt_cat_{region_name}_M{mass}"] = categorizer(
            copy_function(tmp_func, f"bdt_cat_{region_name}_M{mass}"),
            uses=uses,
        )