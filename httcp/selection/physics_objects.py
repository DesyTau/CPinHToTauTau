# coding: utf-8

"""
Exemplary selection methods.
"""

from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.util import sorted_indices_from_mask
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
) -> tuple[ak.Array, SelectionResult, ak.Array, ak.Array, ak.Array]:
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

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    muon_mask  = ak.local_index(events.Muon.pt) >= 0
    
    good_muon_mask = muon_mask
    single_veto_muon_mask = muon_mask
    double_veto_muon_mask = muon_mask
    selection_steps = {}

    for cut in good_selections.keys():
        good_muon_mask = good_muon_mask & good_selections[cut]
        selection_steps[cut] = good_muon_mask


    for cut in single_veto_selections.keys():
        single_veto_muon_mask = single_veto_muon_mask & single_veto_selections[cut]
    #single_veto_muon_mask = single_veto_muon_mask & ~good_muon_mask

    for cut in double_veto_selections.keys():
        double_veto_muon_mask = double_veto_muon_mask & double_veto_selections[cut]
    #double_veto_muon_mask = double_veto_muon_mask & ~good_muon_mask

    # convert to sorted indices
    good_muon_indices = sorted_indices[good_muon_mask[sorted_indices]]
    good_muon_indices = ak.values_astype(good_muon_indices, np.int32)

    veto_muon_indices = sorted_indices[single_veto_muon_mask[sorted_indices]]
    veto_muon_indices = ak.values_astype(veto_muon_indices, np.int32)

    double_veto_muon_indices = sorted_indices[double_veto_muon_mask[sorted_indices]]
    double_veto_muon_indices = ak.values_astype(double_veto_muon_indices, np.int32)

    return events, SelectionResult(
        aux=selection_steps,
    ), good_muon_indices, veto_muon_indices, double_veto_muon_indices


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
) -> tuple[ak.Array, SelectionResult, ak.Array, ak.Array, ak.Array]:
    """
    Electron selection returning two sets of indidces for default and veto muons.
    
    References:
      - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    # >= nano v10
    mva_iso_wp80 = events.Electron.mvaIso_WP80
    mva_iso_wp90 = events.Electron.mvaIso_WP90
    mva_noniso_wp90 = events.Electron.mvaNoIso_WP90

    good_selections = {
        "electron_pt_25"          : events.Electron.pt > 25,
        "electron_eta_2p1"        : abs(events.Electron.eta) < 2.1,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_mva_iso_wp80"   : mva_iso_wp80 == 1,
    }
    single_veto_selections = {
        "electron_pt_10"          : events.Electron.pt > 10,
        "electron_eta_2p5"        : abs(events.Electron.eta) < 2.5,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_mva_noniso_wp90": mva_noniso_wp90 == 1,
        "electron_convVeto"       : events.Electron.convVeto == 1,
        #"electron_lostHits"       : events.Electron.lostHits <= 1,
        "electron_pfRelIso03_all" : events.Electron.pfRelIso03_all < 0.3,
    }
    double_veto_selections = {
        "electron_pt_15"          : events.Electron.pt > 15,
        "electron_eta_2p5"        : abs(events.Electron.eta) < 2.5,
        "electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "electron_cutBased"       : events.Electron.cutBased == 1,
        "electron_pfRelIso03_all" : events.Electron.pfRelIso03_all < 0.3,
    }

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)    
    electron_mask  = ak.local_index(events.Electron.pt) >= 0
    good_electron_mask = electron_mask
    single_veto_electron_mask = electron_mask
    double_veto_electron_mask = electron_mask
    selection_steps = {}

    for cut in good_selections.keys():
        good_electron_mask = good_electron_mask & good_selections[cut]
        selection_steps[cut] = good_electron_mask


    for cut in single_veto_selections.keys():
        single_veto_electron_mask = single_veto_electron_mask & single_veto_selections[cut]
    #single_veto_electron_mask = single_veto_electron_mask & ~good_electron_mask

    for cut in double_veto_selections.keys():
        double_veto_electron_mask = double_veto_electron_mask & double_veto_selections[cut]
    #double_veto_electron_mask = double_veto_electron_mask & ~good_electron_mask

    # convert to sorted indices
    good_electron_indices = sorted_indices[good_electron_mask[sorted_indices]]
    good_electron_indices = ak.values_astype(good_electron_indices, np.int32)

    veto_electron_indices = sorted_indices[single_veto_electron_mask[sorted_indices]]
    veto_electron_indices = ak.values_astype(veto_electron_indices, np.int32)

    double_veto_electron_indices = sorted_indices[double_veto_electron_mask[sorted_indices]]
    double_veto_electron_indices = ak.values_astype(double_veto_electron_indices, np.int32)


    return events, SelectionResult(
        aux=selection_steps,
    ), good_electron_indices, veto_electron_indices, double_veto_electron_indices


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
        electron_indices: ak.Array,
        muon_indices: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult, ak.Array]:
    """
    Tau selection returning two sets of indidces for default and veto muons.
    
    References:
      - 
    """
    # https://cms-nanoaod-integration.web.cern.ch/integration/cms-swmaster/data106Xul17v2_v10_doc.html#Tau
    tau_vs_e = DotDict(vvloose=2, vloose=3)
    tau_vs_mu = DotDict(vloose=1, tight=4)
    tau_vs_jet = DotDict(vvloose=2, loose=4, medium=5)
    
    good_selections = {
        "tau_pt_20"     : events.Tau.pt > 20,
        "tau_eta_2p3"   : abs(events.Tau.eta) < 2.3,
        "tau_dz_0p2"    : abs(events.Tau.dz) < 0.2,
        "DeepTauVSjet"  : events.Tau.idDeepTau2018v2p5VSjet >= tau_vs_jet.medium,
        "DeepTauVSe"    : events.Tau.idDeepTau2018v2p5VSe   >= tau_vs_e.vvloose,
        "DeepTauVSmu"   : events.Tau.idDeepTau2018v2p5VSmu  >= tau_vs_mu.tight,
        #"CleanFromEle"  : ak.all(events.Tau.metric_table(events.Electron[electron_indices]) > 0.5, axis=2),
        #"CleanFromMu"   : ak.all(events.Tau.metric_table(events.Muon[muon_indices]) > 0.5, axis=2),
    }

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Tau.pt, axis=-1, ascending=False)
    tau_mask  = ak.local_index(events.Tau.pt) >= 0

    good_tau_mask = tau_mask
    selection_steps = {}

    for cut in good_selections.keys():
        good_tau_mask = good_tau_mask & good_selections[cut]
        selection_steps[cut] = good_tau_mask

    # convert to sorted indices
    good_tau_indices = sorted_indices[good_tau_mask[sorted_indices]]
    good_tau_indices = ak.values_astype(good_tau_indices, np.int32)

    return events, SelectionResult(
        aux=selection_steps,
    ), good_tau_indices


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
    sorted_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)

    # nominal selection
    good_selections = {
        "jet_pt_30"               : events.Jet.pt > 30.0,
        "jet_eta_2.4"             : abs(events.Jet.eta) < 2.4,
        "jet_id"                  : events.Jet.jetId == 6,  # tight plus lepton veto
        "jet_puId"                : ((events.Jet.pt >= 50.0) 
                                     | (events.Jet.puId == (1 if is_2016 else 4)))
    }
    
    jet_mask  = ak.local_index(events.Jet.pt) >= 0
    
    good_jet_mask = jet_mask
    selection_steps = {}

    for cut in good_selections.keys():
        good_jet_mask = good_jet_mask & good_selections[cut]
        selection_steps[cut] = good_jet_mask
        #selection_steps[cut] = ak.sum(good_jet_mask, axis=1) > 0

    # b-tagged jets, tight working point
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (good_jet_mask) & (events.Jet.btagDeepFlavB >= wp_tight)

    good_jet_indices = sorted_indices[good_jet_mask[sorted_indices]]
    good_jet_indices = ak.values_astype(good_jet_indices, np.int32)

    # bjet veto
    bjet_veto = ak.sum(bjet_mask, axis=1) == 0

    return events, SelectionResult(
        steps = {
            "b_veto": bjet_veto,
        }, 
        objects = {
            "Jet": {
                "Jet": good_jet_indices,
            },
        },
        aux = selection_steps,
    )
    
    
"""
@selector(
    uses={
        "GenPart.*",
    },
    exposed=False
)
def gentau_selection(
        self: Selector,
        events: ak.Array,
        **kwargs,
) -> ak.Array:
    genpart = events.GenPart
    istau_mask = (np.abs(genpart.pdgId) == 15) & (genpart.status == 2)
    
    return istau_mask
"""
