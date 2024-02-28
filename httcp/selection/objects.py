# coding: utf-8

"""
Exemplary selection methods.
"""

from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import, DotDict

from httcp.util import IF_NANO_V9, IF_NANO_V11

np = maybe_import("numpy")
ak = maybe_import("awkward")


# ------------------------------------------------------------------------------------------------------- #
# Muon Selection
# Reference:
#   https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
#   http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
# ------------------------------------------------------------------------------------------------------- #
@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.dxy", "Muon.dz", "Muon.mediumId", 
        "Muon.pfRelIso04_all", "Muon.isGlobal", "Muon.isPFcand", 
        #"Muon.isTracker",
    },
    exposed=False,
)
def muon_selection(
        self: Selector,
        events: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    """
    Muon selection returning two sets of indidces for default and veto muons.
    
    References:
      - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
      - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15
    """
    good_selections = {
        "muon_pt_26"          : events.Muon.pt > 26,
        "muon_eta_2p4"        : abs(events.Muon.eta) < 2.4,
        "mediumID"            : events.Muon.mediumId == 1,
        "muon_dxy_0p045"      : abs(events.Muon.dxy) < 0.045,
        "muon_dz_0p2"         : abs(events.Muon.dz) < 0.2,
        "muon_iso_0p15"       : events.Muon.pfRelIso04_all < 0.15
    }
    single_veto_selections = {
        "muon_pt_10"          : events.Muon.pt > 10,
        "muon_eta_2p4"        : abs(events.Muon.eta) < 2.4,
        "mediumID"            : events.Muon.mediumId == 1,
        "muon_dxy_0p045"      : abs(events.Muon.dxy) < 0.045,
        "muon_dz_0p2"         : abs(events.Muon.dz) < 0.2,
        "muon_iso_0p3"        : events.Muon.pfRelIso04_all < 0.3
    }
    double_veto_selections = {
        "muon_pt_15"          : events.Muon.pt > 15,
        "muon_eta_2p4"        : abs(events.Muon.eta) < 2.4,
        "muon_isGlobal"       : events.Muon.isGlobal == True,
        "muon_isPF"           : events.Muon.isPFcand == True,
        #"muon_isTracker"      : events.Muon.isTracker ==True,
        "muon_dxy_0p045"      : abs(events.Muon.dxy) < 0.045,
        "muon_dz_0p2"         : abs(events.Muon.dz) < 0.2,
        "muon_iso_0p3"        : events.Muon.pfRelIso04_all < 0.3
    }
    
    muon_mask  = ak.local_index(events.Muon.pt) >= 0
    
    good_muon_mask = ak.copy(muon_mask)
    single_veto_muon_mask = ak.copy(muon_mask)
    double_veto_muon_mask = ak.copy(muon_mask)
    selection_steps = {}

    for cut in good_selections.keys():
        good_muon_mask = good_muon_mask & good_selections[cut]
        selection_steps[cut] = ak.sum(good_selections[cut], axis=1) > 0

    for cut in single_veto_selections.keys():
        single_veto_muon_mask = single_veto_muon_mask & single_veto_selections[cut]

    for cut in double_veto_selections.keys():
        double_veto_muon_mask = double_veto_muon_mask & double_veto_selections[cut]


    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Muon": {"MySelectedMuon": indices_applied_to_Muon}}
    return events, SelectionResult(
        steps=selection_steps,
        objects={
            "Muon": {
                "Muon": good_muon_mask,
                "VetoMuon": single_veto_muon_mask & ~good_muon_mask,
                "DoubleVetoMuon": double_veto_muon_mask,
            },
        },
    )


# ------------------------------------------------------------------------------------------------------- #
# Electron Selection
# Reference:
#   https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
#   http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
# ------------------------------------------------------------------------------------------------------- #
@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.dxy", "Electron.dz",
        "Electron.pfRelIso03_all", "Electron.convVeto", #"Electron.lostHits",
        IF_NANO_V9("Electron.mvaFall17V2Iso_WP80", "Electron.mvaFall17V2Iso_WP90", "Electron.mvaFall17V2noIso_WP90"),
        IF_NANO_V11("Electron.mvaIso_WP80", "Electron.mvaIso_WP90", "Electron.mvaNoIso_WP90"),
        "Electron.cutBased",
    },
    exposed=False,
)
def electron_selection(
        self: Selector,
        events: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    """
    Electron selection returning two sets of indidces for default and veto muons.
    
    References:
      - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
        mva_noniso_wp90 = events.Electron.mvaNoIso_WP90
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        mva_noniso_wp90 = events.Electron.mvaFall17V2noIso_WP90

    good_selections = {
        "electron_pt_26"          : events.Electron.pt > 26,
        "electron_eta_2p1"        : abs(events.Electron.eta) < 2.1,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_mva_iso_wp80"   : mva_iso_wp80 == 1
    }
    single_veto_selections = {
        "electron_pt_10"          : events.Electron.pt > 10,
        "electron_eta_2p5"        : abs(events.Electron.eta) < 2.5,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_mva_noniso_wp90": mva_noniso_wp90 == 1,
        "electron_convVeto"       : events.Electron.convVeto == 1,
        #"electron_lostHits"       : events.Electron.lostHits <= 1,
        "electron_pfRelIso03_all" : events.Electron.pfRelIso03_all < 0.3
    }
    double_veto_selections = {
        "electron_pt_15"          : events.Electron.pt > 15,
        "electron_eta_2p5"        : abs(events.Electron.eta) < 2.5,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_cutBased"       : events.Electron.cutBased == 1,
        "electron_pfRelIso03_all" : events.Electron.pfRelIso03_all < 0.3
    }
    
    electron_mask  = ak.local_index(events.Electron.pt) >= 0
    
    good_electron_mask = ak.copy(electron_mask)
    single_veto_electron_mask = ak.copy(electron_mask)
    double_veto_electron_mask = ak.copy(electron_mask)
    selection_steps = {}

    for cut in good_selections.keys():
        good_electron_mask = good_electron_mask & good_selections[cut]
        selection_steps[cut] = ak.sum(good_selections[cut], axis=1) > 0

    for cut in single_veto_selections.keys():
        single_veto_electron_mask = single_veto_electron_mask & single_veto_selections[cut]

    for cut in double_veto_selections.keys():
        double_veto_electron_mask = double_veto_electron_mask & double_veto_selections[cut]


    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Electron": {"MySelectedElectron": indices_applied_to_Electron}}
    return events, SelectionResult(
        steps=selection_steps,
        objects={
            "Electron": {
                "Electron": good_electron_mask,
                "VetoElectron": single_veto_electron_mask & ~good_electron_mask,
                "DoubleVetoElectron": double_veto_electron_mask,
            },
        },
    )


# ------------------------------------------------------------------------------------------------------- #
# Tau Selection
# Reference:
#   https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
#   http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
# ------------------------------------------------------------------------------------------------------- #
@selector(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.dz", 
        "Tau.idDeepTau2018v2p5VSe",
        "Tau.idDeepTau2018v2p5VSmu", 
        "Tau.idDeepTau2018v2p5VSjet",
    },
    exposed=False,
)
def tau_selection(
        self: Selector,
        events: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    """
    Tau selection returning two sets of indidces for default and veto muons.
    
    References:
      - 
    """
    if self.config_inst.campaign.x.version < 10:
        # https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html
        tau_vs_e = DotDict(vvloose=2, vloose=4)
        tau_vs_mu = DotDict(vloose=1, tight=8)
        tau_vs_jet = DotDict(vvloose=2, loose=8, medium=16)
    else:
        # https://cms-nanoaod-integration.web.cern.ch/integration/cms-swmaster/data106Xul17v2_v10_doc.html#Tau
        tau_vs_e = DotDict(vvloose=2, vloose=3)
        tau_vs_mu = DotDict(vloose=1, tight=4)
        tau_vs_jet = DotDict(vvloose=2, loose=4, medium=5)
        
    good_selections = {
        "tau_pt_20"     : events.Tau.pt > 20
        "tau_eta_2p3"   : abs(events.Tau.eta) < 2.3,
        "tau_dz_0p2"    : abs(events.Tau.dz) < 0.2,
        "DeepTauVSjet"  : events.Tau.idDeepTau2018v2p5VSjet >= tau_vs_jet.medium,
        "DeepTauVSe"    : events.Tau.idDeepTau2018v2p5VSe   >= tau_vs_e.vvloose,
        "DeepTauVSmu"   : events.Tau.idDeepTau2018v2p5VSmu  >= tau_vs_mu.tight,
    }

    tau_mask  = ak.local_index(events.Tau.pt) >= 0
    
    good_tau_mask = ak.copy(tau_mask)
    selection_steps = {}

    for cut in good_selections.keys():
        good_tau_mask = good_tau_mask & good_selections[cut]
        selection_steps[cut] = ak.sum(good_selections[cut], axis=1) > 0

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Muon": {"MySelectedMuon": indices_applied_to_Muon}}
    return events, SelectionResult(
        steps=selection_steps,
        objects={
            "Tau": {
                "Tau": good_tau_mask,
            },
        },
    )


# ------------------------------------------------------------------------------------------------------- #
# Jet Selection
# Reference:
#   https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
#   http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
# ------------------------------------------------------------------------------------------------------- #
@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.jetId", "Jet.puId", "Jet.btagDeepFlavB"
    },
    exposed=False,
)
def jet_selection(
        self: Selector,
        events: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    """
    Tau selection returning two sets of indidces for default and veto muons.
    
    References:
      - 
    """
    is_2016 = self.config_inst.campaign.x.year == 2016

    # nominal selection
    good_selections = {
        "jet_pt_30"               : events.Jet.pt > 30.0,
        "jet_eta_2.4"             : abs(events.Jet.eta) < 2.4,
        "jet_id"                  : events.Jet.jetId == 6,  # tight plus lepton veto
        "jet_puId"                : ((events.Jet.pt >= 50.0) 
                                     | (events.Jet.puId == (1 if is_2016 else 4)))
    }
    
    jet_mask  = ak.local_index(events.Jet.pt) >= 0
    
    good_jet_mask = ak.copy(jet_mask)
    selection_steps = {}

    for cut in good_selections.keys():
        good_jet_mask = good_jet_mask & good_selections[cut]
        selection_steps[cut] = ak.sum(good_selections[cut], axis=1) > 0

    # b-tagged jets, tight working point
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (good_jet_mask) & (events.Jet.btagDeepFlavB >= wp_tight)

    # bjet veto
    bjet_veto = ak.sum(bjet_mask, axis=1) == 0

    

    return events, SelectionResult(
        steps={
            "b_jet_veto": bjet_veto
        },
        objects={
            "Jet": {
                "Jet": good_jet_mask,
            },
        },
    )
    
