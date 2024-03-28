# coding: utf-8

"""
Extra-Lepton-Veto
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import, DotDict



np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
    },
    exposed=False,
)
def extra_lepton_veto(
        self: Selector,
        events: ak.Array,
        extra_electron_index: ak.Array,
        extra_muon_index: ak.Array,
        hcand_pair: ak.Array,
        **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    extra_lep  = ak.Array(ak.concatenate([events.Muon[extra_muon_index],
                                          events.Electron[extra_electron_index]], axis=-1),
                          behavior=coffea.nanoevents.methods.nanoaod.behavior)
    extra_lep  = ak.with_name(extra_lep, "PtEtaPhiMLorentzVector")

    has_single_pair = ak.num(hcand_pair.pt, axis=1) == 2
    # keep all True -> [[True], [True], [True], ..., [True]]
    # because, not applying any veto if there is more than one higgs cand pair
    # and keeping those events for now
    dummy = (events.event < 0)[:,None]
    hcand_pair_p4 = ak.firsts(1 * hcand_pair, axis=1)
    hcand_lep1 = hcand_pair_p4[:,:1]
    hcand_lep2 = hcand_pair_p4[:,1:2]

    #print(hcand_lep1.pt, hcand_lep2.pt)
    #print("aksnjklansldjknalknxlk")
    dr_hlep1_extraleps = extra_lep.metric_table(hcand_lep1)
    dr_hlep2_extraleps = extra_lep.metric_table(hcand_lep2)

    dr_mask = (
        ((dr_hlep2_extraleps > 0.5) 
         &  (dr_hlep1_extraleps > 0.001)) 
        | (dr_hlep1_extraleps > 0.5))
    
    has_extra_lepton = ak.where(has_single_pair, 
                                ak.any(dr_mask, axis=-1),
                                dummy)

    has_no_extra_lepton = ak.sum(has_extra_lepton, axis=1) == 0

    return events, SelectionResult(steps={"extra_lepton_veto": has_no_extra_lepton})



@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.charge",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.charge",
    },
    exposed=False,
)
def double_lepton_veto(
        self: Selector,
        events: ak.Array,
        double_veto_electron_index: ak.Array,
        double_veto_muon_index: ak.Array,
        **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    double_veto_muon     = ak.Array(events.Muon[double_veto_muon_index], 
                                    behavior=coffea.nanoevents.methods.nanoaod.behavior)

    double_veto_muon     = ak.with_name(double_veto_muon, "PtEtaPhiMLorentzVector")

    double_veto_electron = ak.Array(events.Electron[double_veto_electron_index], 
                                    behavior=coffea.nanoevents.methods.nanoaod.behavior)
    double_veto_electron = ak.with_name(double_veto_electron, "PtEtaPhiMLorentzVector")

    mu_pair = ak.combinations(double_veto_muon, 2, axis=1)
    el_pair = ak.combinations(double_veto_electron, 2, axis=1)

    mu1,mu2 = ak.unzip(mu_pair)
    el1,el2 = ak.unzip(el_pair)

    presel_mask = lambda leps1, leps2: ((leps1.charge * leps2.charge < 0) & (leps1.delta_r(leps2) > 0.15))

    dlveto_mu_mask = presel_mask(mu1,mu2)
    dlveto_el_mask = presel_mask(el1,el2)

    dl_mu_veto = ak.sum(dlveto_mu_mask, axis=1) == 0
    dl_el_veto = ak.sum(dlveto_el_mask, axis=1) == 0

    dl_veto    = dl_mu_veto & dl_el_veto

    return events, SelectionResult(steps={"dilepton_veto": dl_veto})
    

