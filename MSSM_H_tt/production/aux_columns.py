"""
Produce channel_id column. This function is called in the main selector
"""

from columnflow.production import Producer, producer
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.util import maybe_import, DotDict
from MSSM_H_tt.util import get_lep_p4, get_vec_p3, to_pt_eta_phi_m

np = maybe_import("numpy")
ak = maybe_import("awkward")

import functools
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

# standard CMS-style coefficient used in D_zeta
DZETA_VIS_COEFF = 0.85


def _wrap_delta_phi(dphi):
    dphi = ak.where(dphi > np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
    return dphi
def _safe_transverse_mass(pt1, phi1, pt2, phi2, default=EMPTY_FLOAT):
    """
    Compute sqrt(2 * pt1 * pt2 * (1 - cos(delta_phi))).

    ColumnFlow output columns must be finite, so invalid pairs are filled
    with EMPTY_FLOAT instead of NaN.
    """
    dphi = _wrap_delta_phi(phi1 - phi2)
    arg = 2.0 * pt1 * pt2 * (1.0 - np.cos(dphi))

    valid = (
        ak.is_valid(pt1)
        & ak.is_valid(phi1)
        & ak.is_valid(pt2)
        & ak.is_valid(phi2)
        & (pt1 > 0.0)
        & (pt2 > 0.0)
        & (arg >= -1e-9)
    )
    valid = ak.fill_none(valid, False)

    arg_safe = ak.where(arg > 0.0, arg, 0.0)
    arg_safe = ak.fill_none(arg_safe, 0.0)

    mt = np.sqrt(arg_safe)

    return ak.where(
        valid,
        mt,
        ak.full_like(mt, default),
    )
    
def _clean_jets(self, events):
    """
    Clean jets against the two selected leptons and apply the same final jet
    acceptance used in jet_pt_def.
    """
    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]

    abs_eta = np.abs(sorted_jets.eta)
    mask = (
        sorted_jets.pass_tightID_lep_veto
        & (abs_eta < 4.7)
        & (
            ((sorted_jets.pt > 20.0) & (abs_eta <= 2.5))
            | ((sorted_jets.pt > 50.0) & (abs_eta > 2.5) & (abs_eta <= 3.0))
            | ((sorted_jets.pt > 30.0) & (abs_eta > 3.0) & (abs_eta < 4.7))
        )
    )
    hcand = events["hcand_emu"] 
    for lep_str in ["lep0", "lep1"]:
        lep = ak.firsts(hcand[lep_str])
        seed_idx = ak.fill_none(lep.jetIdx, -1)
        mask = mask & (jet_pt_sorted_idx != seed_idx)

        dphi = _wrap_delta_phi(sorted_jets.phi - lep.phi)
        deta = sorted_jets.eta - lep.eta
        dr = np.sqrt(dphi**2 + deta**2)
        mask = mask & ak.fill_none(dr > 0.4, True)

    return ak.drop_none(ak.mask(sorted_jets, mask))


def _clean_taggable_jets(self, events):
    """
    Jets usable for b-tag-related observables, without applying a b-tag cut.
    """
    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]

    mask = (
        (sorted_jets.pt > 20.0)
        & (np.abs(sorted_jets.eta) < 2.5)
        & sorted_jets.pass_tightID_lep_veto
    )
    hcand = events["hcand_emu"] 
    for lep_str in ["lep0", "lep1"]:
        lep = ak.firsts(hcand[lep_str])
        seed_idx = ak.fill_none(lep.jetIdx, -1)
        mask = mask & (jet_pt_sorted_idx != seed_idx)

        dphi = _wrap_delta_phi(sorted_jets.phi - lep.phi)
        deta = sorted_jets.eta - lep.eta
        dr = np.sqrt(dphi**2 + deta**2)
        mask = mask & ak.fill_none(dr > 0.4, True)

    return ak.drop_none(ak.mask(sorted_jets, mask))


def _clean_bjets(self, events, wp_name="medium"):
    """
    Clean b-jets against the two selected leptons and apply ParticleNet WP.
    """
    year = self.config_inst.x.year
    tag = self.config_inst.x.tag
    particle_net_wps = self.config_inst.x.btag_working_points[year][tag].particleNet
    btag_wp = getattr(particle_net_wps, wp_name)

    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]

    mask = (
        (sorted_jets.pt > 20.0)
        & (np.abs(sorted_jets.eta) < 2.5)
        & sorted_jets.pass_tightID_lep_veto
        & (sorted_jets.btagPNetB >= btag_wp)
    )
    hcand = events["hcand_emu"]
    for lep_str in ["lep0", "lep1"]:
        lep = ak.firsts(hcand[lep_str])
        seed_idx = ak.fill_none(lep.jetIdx, -1)
        mask = mask & (jet_pt_sorted_idx != seed_idx)

        dphi = _wrap_delta_phi(sorted_jets.phi - lep.phi)
        deta = sorted_jets.eta - lep.eta
        dr = np.sqrt(dphi**2 + deta**2)
        mask = mask & ak.fill_none(dr > 0.4, True)

    return ak.drop_none(ak.mask(sorted_jets, mask))


def _pair_object_with_lepton_mass_and_dr(objects, lep):
    """
    objects: jagged collection per event
    lep:     single lepton per event (possibly None)
    """
    pairs = ak.cartesian([objects, ak.singletons(lep)], axis=1)
    obj_br, lep_br = ak.unzip(pairs)
    obj_p4 = get_lep_p4(obj_br)
    lep_p4 = get_lep_p4(lep_br)
    m = (obj_p4 + lep_p4).mass
    dr = obj_p4.delta_r(lep_p4)
    return m, dr


@producer(
    produces={
        "channel_id",
    },
    exposed=False,
)
def channel_id(
        self: Producer,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    channel_id = ak.zeros_like(ak.local_index(events.event), dtype=np.uint8)

    and_mask = ak.ones_like(ak.local_index(events.event), dtype=np.bool_)

    for channel in self.config_inst.channels.names():
        the_mask = ak.num(events[f"hcand_{channel}"].lep0, axis=1) > 0
        and_mask = and_mask & the_mask
        channel_id = ak.where(
            the_mask,
            self.config_inst.get_channel(channel).id,
            channel_id,
        )

    channel_id = ak.values_astype(channel_id, np.uint8)
    events = set_ak_column(events, "channel_id", channel_id)

    return events
@producer(
    uses={
        "Jet.eta",
        "Jet.jetId",
        "Jet.neHEF",
        "Jet.neEmEF",
        "Jet.chMultiplicity",
        "Jet.neMultiplicity",
        "Jet.chHEF",
        "Jet.muEF",
        "Jet.chEmEF",
    },
    produces={
        "Jet.pass_tightID",
        "Jet.pass_tightID_lep_veto",
    },
    exposed=False,
)
def create_jetID_masks(
        self: Producer,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    """
    For nanoaod v13 and v14 jetID is bugged, so there is a special procedure to apply Tight jet ID 
    """
    nano_version = self.config_inst.campaign.x.version
    jets = events.Jet
    if nano_version in [13, 14]:
        print(f'Applying custom tightJetID for nanoAOD v{nano_version}...')
        tightID_eta_2p6 = ((jets.neHEF < 0.99)
                           & (jets.neEmEF < 0.9)
                           & ((jets.chMultiplicity + jets.neMultiplicity) > 1)
                           & (jets.chHEF > 0.01)
                           # Tight criteria for |eta| < 2.6
                           & (jets.chMultiplicity > 0))

        # Tight criteria for 2.6 < |eta| <= 2.7
        tightID_eta_2p6_to_2p7 = ((jets.neHEF < 0.9) & (jets.neEmEF < 0.99))

        # Tight criteria for 2.7 < |eta| <= 3.0
        tightID_eta_2p7_to_3p0 = (jets.neHEF < 0.99)

        tightID_eta_geq_3p0 = ((jets.neMultiplicity >= 2) & (
            jets.neEmEF < 0.4))  # Tight criteria for |eta| >= 3.0

        pass_tightID = (((abs(jets.eta) < 2.6) & tightID_eta_2p6)
                        | ((abs(jets.eta) >= 2.6) & (abs(jets.eta) < 2.7) & tightID_eta_2p6_to_2p7)
                        | ((abs(jets.eta) >= 2.7) & (abs(jets.eta) < 3.0) & tightID_eta_2p7_to_3p0)
                        | ((abs(jets.eta) >= 3.0) & tightID_eta_geq_3p0)
                        )

        pass_tightID_lep_veto = ak.where((abs(jets.eta) < 2.7),
                                         pass_tightID & (jets.muEF < 0.8) & (
                                             jets.chEmEF < 0.8),
                                         pass_tightID)
    else:
        pass_tightID_lep_veto = (abs(jets.eta) < 2.7) & ((jets.jetId & 3) != 0)

    jets["pass_tightID"] = pass_tightID
    jets["pass_tightID_lep_veto"] = pass_tightID_lep_veto
    events = set_ak_column(events, "Jet", jets)
    return events

@producer(
    uses={f"Jet.{var}" for var in [
        "pt", "eta", "phi", "mass", "btagDeepFlavB", "btagPNetB", "pass_tightID_lep_veto",
    ]} | {f"hcand_emu.lep0.{var}" for var in [
        "jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",
    ]} | {f"hcand_emu.lep1.{var}" for var in [
        "jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",
    ]},

    produces={
        "n_jets",
        "n_jets_clipped",
        "mt_jets",
        "lead_jet.*",
        "sublead_jet.*",
        "dijet.*",
    },
    exposed=False,
)
def jet_pt_def(
        self: Producer,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    """
    Produce standard cleaned jet observables.

    Produces:
      - n_jets
      - n_jets_clipped
      - mt_jets
      - lead_jet.{pt,eta,phi,mass}
      - sublead_jet.{pt,eta,phi,mass}
      - dijet.{pt,eta,phi,mass,deltaeta,deltaphi,delta_r}
    """
    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]

    jet_selections = {
        "jet_pt_30": sorted_jets.pt > 30.0,
        "jet_eta_4.7": abs(sorted_jets.eta) < 4.7,
        "jet_id": sorted_jets.pass_tightID_lep_veto,
    }

    jet_obj_mask = ak.ones_like(jet_pt_sorted_idx, dtype=np.bool_)
    for the_sel in jet_selections.values():
        jet_obj_mask = jet_obj_mask & the_sel

    ch_str = self.config_inst.channels.names()[0]
    hcand = events[f"hcand_{ch_str}"]

    mask = ak.ones_like(sorted_jets.pt, dtype=np.bool_)
    empty_p4 = ak.zeros_like(get_lep_p4(hcand.lep0))

    for lep_str in ["lep0", "lep1"]:
        lep = ak.firsts(hcand[lep_str])
        seed_idx = ak.fill_none(lep.jetIdx, -1)

        jet_obj_mask_seed_idx = jet_obj_mask & (jet_pt_sorted_idx != seed_idx)
        presel_jet = ak.mask(sorted_jets, jet_obj_mask_seed_idx)

        _jet_br, _lep_br = ak.unzip(ak.cartesian([presel_jet, lep], axis=1))
        jet_p4 = get_lep_p4(_jet_br)
        lep_p4 = get_lep_p4(_lep_br)

        mask = mask & ak.fill_none((jet_p4.delta_r(lep_p4) > 0.4), False)

    jets = get_lep_p4(sorted_jets[mask])

    jet_pt_mask = ((jets.pt > 20) & (abs(jets.eta) <= 2.5))
    jet_pt_mask = jet_pt_mask | (
        (jets.pt > 50)
        & (abs(jets.eta) <= 3.0)
        & (abs(jets.eta) > 2.5)
    )
    jet_pt_mask = jet_pt_mask | (
        (jets.pt > 30)
        & (abs(jets.eta) > 3.0)
    )

    sel_jets = ak.drop_none(ak.mask(jets, jet_pt_mask))
    njets = ak.num(sel_jets, axis=1)

    n_jets_clipped = ak.where(njets > 3, 3, njets)
    n_jets_clipped = ak.values_astype(n_jets_clipped, np.int32)

    lead_jet_p4 = ak.where(njets > 0, sel_jets[:, :1], empty_p4)
    sublead_jet_p4 = ak.where(njets > 1, sel_jets[:, 1:2], empty_p4)
    dijet_p4 = to_pt_eta_phi_m(lead_jet_p4 + sublead_jet_p4)

    mt_jets = _safe_transverse_mass(
        lead_jet_p4.pt,
        lead_jet_p4.phi,
        sublead_jet_p4.pt,
        sublead_jet_p4.phi,
    )

    mt_jets = ak.fill_none(mt_jets, EMPTY_FLOAT)
    mt_jets = ak.nan_to_num(mt_jets, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT)
    
    lead_jet = {}
    lead_jet_mask = lead_jet_p4.pt > 20

    sublead_jet = {}
    sublead_jet_mask = sublead_jet_p4.pt > 20

    dijet = {}
    dijet_mask = dijet_p4.pt > 20

    empty_float = lambda arr: ak.full_like(arr, EMPTY_FLOAT)

    for var in ["pt", "eta", "phi", "mass"]:
        lead_jet[var] = ak.where(
            lead_jet_mask,
            getattr(lead_jet_p4, var),
            empty_float(getattr(lead_jet_p4, var)),
        )

        sublead_jet[var] = ak.where(
            sublead_jet_mask,
            getattr(sublead_jet_p4, var),
            empty_float(getattr(sublead_jet_p4, var)),
        )

        dijet[var] = ak.where(
            dijet_mask,
            getattr(dijet_p4, var),
            empty_float(getattr(dijet_p4, var)),
        )

    for func_name in ["deltaeta", "deltaphi", "delta_r"]:
        func = getattr(lead_jet_p4, func_name)
        dijet[func_name] = ak.where(
            njets >= 2,
            func(sublead_jet_p4),
            empty_float(dijet_p4.pt),
        )

    events = set_ak_column(events, "lead_jet", ak.zip(lead_jet))
    events = set_ak_column(events, "sublead_jet", ak.zip(sublead_jet))
    events = set_ak_column(events, "dijet", ak.zip(dijet))

    events = set_ak_column(events, "n_jets", njets)
    events = set_ak_column_i32(events, "n_jets_clipped", n_jets_clipped)
    events = set_ak_column_f32(events, "mt_jets", mt_jets)

    return events


@producer(
    uses={f"Jet.{var}" for var in [
        "pt", "eta", "phi", "mass", "btagDeepFlavB", "btagPNetB", "pass_tightID_lep_veto",
    ]} | {f"hcand_emu.lep0.{var}" for var in ["jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",
    ]} | {f"hcand_emu.lep1.{var}" for var in ["jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",]},
    produces={"n_jets_tag"},
    exposed=False,
)
def jets_taggable(
        self: Producer,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    """
    Produce the number of taggable jets.
    """
    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]
    jet_selections = {
        "jet_pt_20": sorted_jets.pt > 20.0,
        "jet_eta_2.5": abs(sorted_jets.eta) < 2.5,
        "jet_id": sorted_jets.pass_tightID_lep_veto,
    }
    jet_obj_mask = ak.ones_like(jet_pt_sorted_idx, dtype=np.bool_)
    for the_sel in jet_selections.values():
        jet_obj_mask = jet_obj_mask & the_sel

    for the_ch in self.config_inst.channels.names():
        hcand = events[f"hcand_{the_ch}"]

        for lep_str in [field for field in hcand.fields if "lep" in field]:
            lep = ak.firsts(hcand[lep_str])
            seed_idx = ak.fill_none(lep.jetIdx, -1)
            jet_obj_mask_seed_idx = jet_obj_mask & (jet_pt_sorted_idx != seed_idx)
            presel_jet = ak.drop_none(ak.mask(sorted_jets, jet_obj_mask_seed_idx))
            jet_tau_pairs = ak.cartesian([presel_jet, lep], axis=1)
            jet_br, lep_br = ak.unzip(jet_tau_pairs)
            delta_phi = (jet_br.phi - lep_br.phi)
            delta_phi = ak.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
            delta_phi = ak.where(delta_phi < -np.pi, delta_phi + 2 * np.pi, delta_phi)
            delta_eta = (jet_br.eta - lep_br.eta)
            delta_r = np.sqrt(delta_phi**2 + delta_eta**2)
            if lep_str == "lep0":
                jet_vs_lep0 = jet_br[delta_r > 0.4]
            elif lep_str == "lep1":
                jet_vs_lep1 = jet_br[delta_r > 0.4]

    lep0_max_obj = ak.max(ak.num(jet_vs_lep0.pt))
    lep1_max_obj = ak.max(ak.num(jet_vs_lep1.pt))
    max_len = ak.max([lep0_max_obj, lep1_max_obj])

    jet_vs_lep0_pt = ak.pad_none(jet_vs_lep0.pt, max_len)
    jet_vs_lep0_pt_tag = ak.fill_none(jet_vs_lep0_pt, EMPTY_FLOAT)
    jet_vs_lep1_pt = ak.pad_none(jet_vs_lep1.pt, max_len)
    jet_vs_lep1_pt_tag = ak.fill_none(jet_vs_lep1_pt, EMPTY_FLOAT)

    n_jets_vs_lep0_tag = ak.sum(jet_vs_lep0_pt_tag > 20, axis=1)
    n_jets_vs_lep1_tag = ak.sum(jet_vs_lep1_pt_tag > 20, axis=1)
    n_jets_mask_tag = (n_jets_vs_lep0_tag == n_jets_vs_lep1_tag)
    n_jets_taggable = ak.where(n_jets_mask_tag, n_jets_vs_lep0_tag, 0)
    events = set_ak_column(events, "n_jets_tag", n_jets_taggable)
    return events


@producer(
    uses={f"Jet.{var}" for var in [
        "pt", "eta", "phi", "mass", "btagDeepFlavB", "btagPNetB", "pass_tightID_lep_veto",
    ]} | {f"hcand_emu.lep0.{var}" for var in [
        "jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",
    ]} | {f"hcand_emu.lep1.{var}" for var in [
        "jetIdx", "pt", "eta", "phi", "mass", "ip_sig", "charge",
    ]},
    produces={
        "N_b_jets",
        "n_bjets_clipped",
        "mt_bjets",
        "lead_b_jet.*",
        "sublead_b_jet.*",
        "di_b_jet.*",
    },
    exposed=False,
)
def number_b_jet(
        self: Producer,
        events: ak.Array,
        **kwargs
) -> ak.Array:
    """
    Produces:
      - N_b_jets
      - n_bjets_clipped
      - mt_bjets
      - lead_b_jet.{pt,eta,phi,mass}
      - sublead_b_jet.{pt,eta,phi,mass}
      - di_b_jet.{pt,eta,phi,mass,deltaeta,deltaphi,delta_r}
    """
    year = self.config_inst.x.year
    tag = self.config_inst.x.tag
    btag_wp = self.config_inst.x.btag_working_points[year][tag].particleNet.medium

    jet_pt_sorted_idx = ak.argsort(events.Jet.pt, axis=1, ascending=False)
    sorted_jets = events.Jet[jet_pt_sorted_idx]

    jet_selections = {
        "jet_pt_20": sorted_jets.pt > 20.0,
        "jet_eta_2.5": abs(sorted_jets.eta) < 2.5,
        "jet_id": sorted_jets.pass_tightID_lep_veto,
        "btag_wp_medium": sorted_jets.btagPNetB >= btag_wp,
    }

    jet_obj_mask = ak.ones_like(jet_pt_sorted_idx, dtype=np.bool_)
    for the_sel in jet_selections.values():
        jet_obj_mask = jet_obj_mask & the_sel

    ch_str = self.config_inst.channels.names()[0]
    hcand = events[f"hcand_{ch_str}"]

    mask = jet_obj_mask

    for lep_str in ["lep0", "lep1"]:
        lep = ak.firsts(hcand[lep_str])
        seed_idx = ak.fill_none(lep.jetIdx, -1)

        mask = mask & (jet_pt_sorted_idx != seed_idx)

        dphi = _wrap_delta_phi(sorted_jets.phi - lep.phi)
        deta = sorted_jets.eta - lep.eta
        dr = np.sqrt(dphi**2 + deta**2)

        mask = mask & ak.fill_none((dr > 0.4), False)

    sel_bjets = ak.drop_none(ak.mask(sorted_jets, mask))
    nbjets = ak.num(sel_bjets, axis=1)

    n_bjets_clipped = ak.where(nbjets > 2, 2, nbjets)
    n_bjets_clipped = ak.values_astype(n_bjets_clipped, np.int32)

    empty_p4 = ak.zeros_like(get_lep_p4(hcand.lep0))
    sel_bjets_p4 = get_lep_p4(sel_bjets)

    lead_bjet_p4 = ak.where(nbjets > 0, sel_bjets_p4[:, :1], empty_p4)
    sublead_bjet_p4 = ak.where(nbjets > 1, sel_bjets_p4[:, 1:2], empty_p4)
    di_bjet_p4 = to_pt_eta_phi_m(lead_bjet_p4 + sublead_bjet_p4)

    mt_bjets = _safe_transverse_mass(
        lead_bjet_p4.pt,
        lead_bjet_p4.phi,
        sublead_bjet_p4.pt,
        sublead_bjet_p4.phi,
    )

    mt_bjets = ak.fill_none(mt_bjets, EMPTY_FLOAT)
    mt_bjets = ak.nan_to_num(mt_bjets, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT)
    
    empty_float = lambda arr: ak.full_like(arr, EMPTY_FLOAT)

    lead_b_jet = {}
    lead_mask = lead_bjet_p4.pt > 20

    sublead_b_jet = {}
    sublead_mask = sublead_bjet_p4.pt > 20

    di_b_jet = {}
    di_mask = di_bjet_p4.pt > 20

    for var in ["pt", "eta", "phi", "mass"]:
        lead_b_jet[var] = ak.where(
            lead_mask,
            getattr(lead_bjet_p4, var),
            empty_float(getattr(lead_bjet_p4, var)),
        )

        sublead_b_jet[var] = ak.where(
            sublead_mask,
            getattr(sublead_bjet_p4, var),
            empty_float(getattr(sublead_bjet_p4, var)),
        )

        di_b_jet[var] = ak.where(
            di_mask,
            getattr(di_bjet_p4, var),
            empty_float(getattr(di_bjet_p4, var)),
        )

    for func_name in ["deltaeta", "deltaphi", "delta_r"]:
        func = getattr(lead_bjet_p4, func_name)
        di_b_jet[func_name] = ak.where(
            nbjets >= 2,
            func(sublead_bjet_p4),
            empty_float(di_bjet_p4.pt),
        )

    events = set_ak_column(events, "lead_b_jet", ak.zip(lead_b_jet))
    events = set_ak_column(events, "sublead_b_jet", ak.zip(sublead_b_jet))
    events = set_ak_column(events, "di_b_jet", ak.zip(di_b_jet))

    events = set_ak_column(events, "N_b_jets", nbjets)
    events = set_ak_column_i32(events, "n_bjets_clipped", n_bjets_clipped)
    events = set_ak_column_f32(events, "mt_bjets", mt_bjets)

    return events

    