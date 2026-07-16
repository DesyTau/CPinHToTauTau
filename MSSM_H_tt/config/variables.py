### This config is used for listing the variables used in the analysis ###

from columnflow.config_util import add_category

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

from columnflow.util import maybe_import
np = maybe_import("numpy")

import json
from pathlib import Path

# BDT adaptive-bin edge files produced by the training post-processing.
# This must match the OUTPUT_BASE used when creating the score tables and
# running the common low/high-mass adaptive rebinning step.
#
# BDT adaptive edges.  This path is intentionally explicit because the 2D
# y-merged binning files are produced under this training-output directory.
# Change only this constant if you switch to another training output, e.g.
# bdt_3_classes_10_features_rawJetCounts.
BDT_OUTPUT_BASE = Path(
    "/eos/project/d/desytau/public/jmalvaso/"
    "bdt_3_classes_10_features_clippedJetCounts"
)
BDT_ADAPTIVE_TAG = "combined_crossApplied"
BDT_DEFAULT_SCORE_BINNING = (30, 0.0, 1.0)

# Location of the irregular/y-merged 2D binning JSON files created by
# adaptive_plot_2d_discriminant_pair(...).
BDT_2D_ADAPTIVE_SUBDIR = "independent_adaptive"
BDT_2D_YMERGED_TOKEN = "yMergedMinWeightedBkg"


def _bdt_mass_region_name(mass: int) -> str:
    mass = int(mass)
    if mass <= 250:
        return "lowMass_Mle250"
    return "highMass_Mgt250"


def _bdt_common_edges_path(mass: int, discriminant: str) -> Path:
    region_name = _bdt_mass_region_name(mass)
    return (
        BDT_OUTPUT_BASE
        / "adaptive_discriminant_rebinning"
        / f"common_{region_name}"
        / discriminant
        / f"{discriminant}_common_region_edges_{region_name}_{BDT_ADAPTIVE_TAG}.json"
    )


def _bdt_per_mass_edges_path(mass: int, discriminant: str) -> Path:
    mass = int(mass)
    return (
        BDT_OUTPUT_BASE
        / f"M{mass}"
        / "adaptive_discriminant_rebinning"
        / f"M{mass}"
        / discriminant
        / f"{discriminant}_adaptive_edges_M{mass}_{BDT_ADAPTIVE_TAG}.json"
    )


def _extract_edges_from_bdt_json(data):
    """
    The common-region JSON stores edges at top level:
        {"edges": [...], ...}

    The per-mass JSON stores them inside:
        {"binning_summary": {"edges": [...]}, ...}
    """
    if isinstance(data, dict):
        if "edges" in data:
            return data["edges"]
        if "binning_summary" in data and "edges" in data["binning_summary"]:
            return data["binning_summary"]["edges"]
    raise KeyError("Could not find adaptive bin edges in JSON payload")


def _read_bdt_adaptive_binning(mass: int, discriminant: str):
    """
    Return adaptive bin edges for the BDT discriminant variable definition.

    Priority:
      1. common low/high-mass edges from the post-processing step
      2. per-mass adaptive edges, if available
      3. fixed fallback binning, so the config remains importable
    """
    for path in (
        _bdt_common_edges_path(mass, discriminant),
        _bdt_per_mass_edges_path(mass, discriminant),
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
            edges = [float(x) for x in _extract_edges_from_bdt_json(data)]
            if len(edges) >= 2:
                return edges
        except Exception:
            pass

    return BDT_DEFAULT_SCORE_BINNING

def keep_columns(cfg: od.Config) -> None:
    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # TauProds
            "TauProd.*",
            "GenPart.*",
            "GenZ.*",
            # general event info
            "run", "luminosityBlock", "event",
            "PV.npvs","Pileup.nTrueInt","Pileup.nPU","genWeight",
            "LHEWeight.originalXWGTUP", "HTXS_njets*", "LHE_Njets","LHEScaleWeight*", "PSWeight*",
            "weight","zpt_weight","muon_weight_nom","mc_weight","tau_weight_nom",
        } | {
            f"PuppiMET.{var}" for var in [
                "pt", "phi", "significance",
                "covXX", "covXY", "covYY",
                "ptUnclusteredUp", "ptUnclusteredDown", 
                "phiUnclusteredUp", "phiUnclusteredDown", 
            ]
        } | {
            f"MET.{var}" for var in [
                "pt", "phi", "significance",
                "covXX", "covXY", "covYY",
            ]     
        } | {
            f"Jet.{var}" for var in [
                "pt", "eta", "phi", "mass", "jetId", 
                "btagDeepFlavB", "hadronFlavour", "btagPNetB",
                "neEmEF","chHEF","neHEF",
                "chHEF","muEF","chEmEF",
                "neMultiplicity","chMultiplicity",
            ] 
        } | {
            f"Tau.{var}" for var in [
                "pt","eta","phi","mass","dxy","dz", "charge", 
                "rawDeepTau2018v2p5VSjet","idDeepTau2018v2p5VSjet", "idDeepTau2018v2p5VSe", "idDeepTau2018v2p5VSmu", 
                "decayMode", "decayModePNet", "genPartFlav", "rawIdx",
                "pt_no_tes", "mass_no_tes", "IPx", "IPy", "IPz","ip_sig", "jetIdx"
            ] 
        } | {
            f"Muon.{var}" for var in [
                "pt","eta","phi","mass","dxy","dz", "charge",
                "decayMode", "pfRelIso04_all","mT", "rawIdx","IPx", "IPy", "IPz","ip_sig", "jetIdx",
            ] 
        } | {
            f"Electron.{var}" for var in [
                "pt","eta","phi","mass","dxy","dz", "charge", 
                "decayMode", "pfRelIso03_all", "mT", "rawIdx", "IPx", "IPy", "IPz","ip_sig", "jetIdx",
                "pt_no_scaling_smearing",
            ] 
        } | {
            f"{var}_triggerd" for var in [ #Trigger variables to have a track of a particular trigger fired
                "single_electron", "cross_electron",
                "single_muon", "cross_muon",
                "cross_tau",
            ]
        } | {
            f"matched_triggerID_{var}" for var in [
                "e", "mu", "tau",
            ]
        } | {
            f"TrigObj.{var}" for var in [
                "id", "pt", "eta", "phi", "filterBits",
            ]
        } | {
            f"TauSpinner.weight_cp_{var}" for var in [
                "0", "0_alt", "0p25", "0p25_alt", "0p375",
                "0p375_alt", "0p5", "0p5_alt", "minus0p25", "minus0p25_alt"
            ]
        } | {
            f"hcand.{var}" for var in [
                "pt","eta","phi","mass", "charge", 
                "decayMode", "rawIdx", "ip_sig", "IPx", "IPy","IPz"
            ]
        } |{
            "GenTau.*", "GenTauProd.*",

            # jet multiplicities and BDT jet inputs
            "nJet",
            "n_jets",
            "N_b_jets",
            "n_jets_clipped",
            "n_bjets_clipped",
            "mt_jets",
            "mt_bjets",

            # jet and b-jet objects
            "lead_jet.*",
            "sublead_jet.*",
            "dijet.*",
            "n_jets_tag",
            "lead_b_jet.*",
            "sublead_b_jet.*",
            "di_b_jet.*",

            "all_triggers_id",
            "triggerID_e",
            "triggerID_mu",
            "triggerID_tau",
            "LHE.Njets",
            "LHE.NpNLO",
        } | {
            f"hcandprod.{var}" for var in [
                "pt", "eta", "phi", "mass", "charge",
                "pdgId", "tauIdx"
            ]
        } | {
		"hcand_*","tau_decay_prods*", "OC_lepton_veto",
	} | {"is_b_vetoed","channel_id"} | {ColumnCollection.ALL_FROM_SELECTOR},
        "cf.MergeSelectionMasks": {
            "normalization_weight", 
            "cutflow.*", "process_id", "category_ids",
    } | { "bdt_*",},
        "cf.UniteColumns": {
            "*",
        },
    })



def add_common_features(cfg: od.config) -> None:
    """
    Adds common features
    """
    cfg.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="N_events",
        expression="N_events",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )

def add_lepton_features(cfg: od.Config) -> None:
    """
    Adds lepton features only , ex electron_1_pt
    """
    cfg.add_variable(
        name=f"electron_1_pt_no_scaling_smearing",
        expression="Electron.pt_no_scaling_smearing[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 200.),
        unit="GeV",
        x_title= r" Electron $p_{T}$ no scaling or smearing",
    )
    
    for obj in ["Electron", "Muon", "Tau"]:
        for i in range(2):
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_pt",
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 200.),
                unit="GeV",
                x_title=obj + r" $p_{T}$",
            )
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_phi",
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(32, -3.2, 3.2),
                x_title=obj + r" $\phi$",
            )
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_eta",
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(25, -2.5, 2.5),
                x_title=obj + r" $\eta$",
            )
        cfg.add_variable(
            name=f"{obj.lower()}_ip_sig",
            expression=f"{obj}.ip_sig",
            null_value=EMPTY_FLOAT,
            binning=(40, 0.0, 10),
            unit="",
            x_title=obj + r"$\frac{|IP|}{\sigma(IP)}$",
        )


def add_jet_features(cfg: od.Config) -> None:
    """
    Adds jet features only
    """
    cfg.add_variable(
        name="n_jet",
        expression="nJet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
   
    cfg.add_variable(
        name="n_j",
        expression="n_jets",
        binning=(4, 0, 4),
        discrete_x=True,
        x_title="N_jets_pT_20_eta_4_7_Tight",
    )
    
    cfg.add_variable(
        name="N_jets_pT_20_eta_2_5_Tight",
        expression="n_jets_tag",
        binning=(4, 0, 4),
        discrete_x=True,
        x_title="N_jets_pT_20_eta_2_5_Tight",
    )
    cfg.add_variable(
        name="N_b_jets",
        expression="N_b_jets",
        binning=(3, 0, 3),
        discrete_x=True,
        x_title="N_b_jets",
    )
    cfg.add_variable(
            name="n_jets_clipped",
            expression="n_jets_clipped",
            binning=(4, -0.5, 3.5),
            discrete_x=True,
            x_title=r"clipped $N_{\mathrm{jets}}$",
    )

    cfg.add_variable(
            name="n_bjets_clipped",
            expression="n_bjets_clipped",
            binning=(3, -0.5, 2.5),
            discrete_x=True,
            x_title=r"clipped $N_{\mathrm{b\,jets}}$",
    )

    cfg.add_variable(
            name="mt_jets",
            expression="mt_jets",
            null_value=EMPTY_FLOAT,
            binning=(25, 0.0, 500.0),
            unit="GeV",
            x_title=r"$m_{T}(j_{1}, j_{2})$",
    )

    cfg.add_variable(
            name="mt_bjets",
            expression="mt_bjets",
            null_value=EMPTY_FLOAT,
            binning=(25, 0.0, 500.0),
            unit="GeV",
            x_title=r"$m_{T}(b_{1}, b_{2})$",
    )
    cfg.add_variable(
        name="leading_jet_pt",
        expression="lead_jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(15, 30.0, 330.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )        
    cfg.add_variable(
        name="subleading_jet_pt",
        expression="sublead_jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(10, 30.0, 280.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )
    cfg.add_variable(
        name="leading_jet_eta",
        expression="lead_jet.eta",
        null_value=EMPTY_FLOAT,
        binning=(24, -4.7, 4.7),
        x_title="Leading Jet $\\eta$",
    ) 
    cfg.add_variable(
        name="subleading_jet_eta",
        expression="sublead_jet.eta",
        null_value=EMPTY_FLOAT,
        binning=(24, -4.7, 4.7),
        x_title="Subleading Jet $\\eta$",
    ) 
    cfg.add_variable(
        name="leading_jet_phi",
        expression="lead_jet.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2, 3.2),
        x_title="Leading Jet $\\phi$",
    )  
    cfg.add_variable(
        name="subleading_jet_phi",
        expression="sublead_jet.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2, 3.2),
        x_title="Subleading Jet $\\phi$",
    ) 
    cfg.add_variable(
        name="dijet_delta_eta",
        expression="dijet.deltaeta",
        null_value=EMPTY_FLOAT,
        binning=(12,-6,6),
        x_title="$\\Delta \\eta_{jj}$",
    ) 
    cfg.add_variable(
        name="dijet_delta_phi",
        expression="dijet.deltaphi",
        null_value=EMPTY_FLOAT,
        binning=(12,-6,6),
        x_title="$\\Delta \\phi_{jj}$",
    ) 
    cfg.add_variable(
        name="dijet_pt",
        expression="dijet.pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 400.0),
        x_title="$pT_{jj}$",
    ) 
    cfg.add_variable(
        name="dijet_delta_r",
        expression="dijet.delta_r",
        null_value=EMPTY_FLOAT,
        binning=(15,0,5),
        x_title="$\\Delta R_{jj}$",
    ) 
    cfg.add_variable(
        name="mjj",
        expression="dijet.mass",
        null_value=EMPTY_FLOAT,
        binning=(20, 10.0, 410.0),
        unit="GeV",
        x_title=r"$m_{jj}$",
    )
    cfg.add_variable(
        name="leading_b_jet_pt",
        expression="lead_b_jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(15, 30.0, 330.0),
        unit="GeV",
        x_title=r"Leading b jet $p_{T}$",
    )        
    cfg.add_variable(
        name="subleading_b_jet_pt",
        expression="sublead_b_jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(10, 30.0, 280.0),
        unit="GeV",
        x_title=r"Subleading b jet $p_{T}$",
    )
    cfg.add_variable(
        name="leading_b_jet_eta",
        expression="lead_b_jet.eta",
        null_value=EMPTY_FLOAT,
        binning=(12, -2.5, 2.5),
        x_title="Leading b Jet $\\eta$",
    ) 
    cfg.add_variable(
        name="subleading_b_jet_eta",
        expression="sublead_b_jet.eta",
        null_value=EMPTY_FLOAT,
        binning=(12, -2.5, 2.5),
        x_title="Subleading b Jet $\\eta$",
    ) 
    cfg.add_variable(
        name="leading_b_jet_phi",
        expression="lead_b_jet.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2, 3.2),
        x_title="Leading b Jet $\\phi$",
    )  
    cfg.add_variable(
        name="subleading_b_jet_phi",
        expression="sublead_b_jet.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2, 3.2),
        x_title="Subleading b Jet $\\phi$",
    ) 
    cfg.add_variable(
        name="di_b_jet_delta_eta",
        expression="di_b_jet.deltaeta",
        null_value=EMPTY_FLOAT,
        binning=(12,-6,6),
        x_title="$\\Delta \\eta_{bb}$",
    ) 
    cfg.add_variable(
        name="di_b_jet_delta_phi",
        expression="di_b_jet.deltaphi",
        null_value=EMPTY_FLOAT,
        binning=(12,-6,6),
        x_title="$\\Delta \\phi_{bb}$",
    ) 
    cfg.add_variable(
        name="di_b_jet_pt",
        expression="di_b_jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 400.0),
        x_title="$pT_{bb}$",
    ) 
    cfg.add_variable(
        name="di_b_jet_delta_r",
        expression="di_b_jet.delta_r",
        null_value=EMPTY_FLOAT,
        binning=(15,0,5),
        x_title="$\\Delta R_{bb}$",
    ) 
    cfg.add_variable(
        name="mb_jb_j",
        expression="di_b_jet.mass",
        null_value=EMPTY_FLOAT,
        binning=(20, 10.0, 410.0),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )            
    cfg.add_variable(
        name="ht",
        expression="ht",
        binning=(20, 0.0, 800.0),
        unit="GeV",
        x_title="HT",
    )
    cfg.add_variable(
        name="jet_raw_DeepJetFlavB",
        expression="Jet.btagDeepFlavB",
        null_value=EMPTY_FLOAT,
        binning=(15, 0,1),
        x_title=r"raw DeepJetFlawB",
    )
    cfg.add_variable(
        name="jet_raw_PNetB",
        expression="Jet.btagPNetB",
        null_value=EMPTY_FLOAT,
        binning=(15, 0,1),
        x_title=r"raw PNetB",
    )
    
def add_highlevel_features(cfg: od.Config) -> None:    
    """
    Adds MET and other high-level features
    """
    cfg.add_variable(
        name="met",
        expression="MET.pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 200.0),
        x_title=r"MET",
    )

    cfg.add_variable(
        name="puppi_met_pt",
        expression="PuppiMET.pt",
        null_value=EMPTY_FLOAT,
        binning=(30, 0,300),
        unit="GeV",
        x_title=r"PUPPI MET $p_T$",
    )
    cfg.add_variable(
        name="puppi_met_pt_recoil_corr",
        expression="RecoilCorrMET.pt",
        null_value=EMPTY_FLOAT,
        binning=(30, 0,300),
        unit="GeV",
        x_title=r"RecoilCorrMET $p_T$",
    )
    cfg.add_variable(
        name="puppi_met_phi",
        expression="PuppiMET.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2,3.2),
        x_title=r"PUPPI MET $\phi$",
    )  
    cfg.add_variable(
        name="D_zeta",
        expression="D_zeta",
        null_value=EMPTY_FLOAT,
        binning=(12, -80, 300),
        x_title="$D_{\\zeta}$"
    )
    
    cfg.add_variable(
        name="pt_H",
        expression="pt_H",
        null_value=EMPTY_FLOAT,
        binning=(12,0,250),
        x_title="$p_{T}(H)$"
    )  
    

def add_weight_features(cfg: od.Config) -> None:
    """
    Adds weights
    """
    cfg.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(20, -2, 2),
        x_title="MC weight",
    )
    cfg.add_variable(
        name="pu_weight",
        expression="pu_weight",
        null_value=EMPTY_FLOAT,
        binning=(30, 0,3),
        unit="",
        x_title=r"Pileup weight",
    )
    
    cfg.add_variable(
        name="muon_weight",
        expression="muon_weight_nom",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.5,1.5),
        unit="",
        x_title=r"muon weight",
    )
    
    cfg.add_variable(
        name="tau_weight",
        expression="tau_weight_nom",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.5,1.5),
        unit="",
        x_title=r"tau weight",
    )
    
    for var in ["0", "0p25", "0p375", "0p5", "minus0p25"]:
        
        angle = float(var.replace("minus","-").replace("p", "."))*180
        cfg.add_variable(
        name=f"TauSpinner_weight_cp_{var}",
        expression=f"TauSpinner.weight_cp_{var}",
        null_value=EMPTY_FLOAT,
        binning=(60, -3,3),
        unit="",
        x_title=fr"Tau spinner weight $\Delta \phi$=${angle}^{{\circ}}$",
    )
        

    

def add_cutflow_features(cfg: od.Config) -> None:
    """
    Adds cf features
    """
    cfg.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )


def phi_cp_variables(cfg: od.Config) -> None:
    n_bins_phi_cp = 11
    for the_ch in ['mu_pi', 'mu_rho', 'mu_a1_1pr', "rho_rho","pi_pi"]:
        spitted_str = the_ch.split('_')
        if 'a1' in the_ch: 
            title_str = "\\" + spitted_str[0] + fr" a_1, {spitted_str[2]}"
        else:
             title_str = "\\" + spitted_str[0] + "\\" + spitted_str[1]
        cfg.add_variable(
            name=f"phi_cp_{the_ch}",
            expression=f"phi_cp_{the_ch}",
            null_value=EMPTY_FLOAT,
            binning=(n_bins_phi_cp, 0, 2*np.pi),
            x_title=rf"$\varphi_{{CP}} [{title_str}]$ (rad)",
        )
        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg1",
            expression=f"phi_cp_{the_ch}_reg1",
            null_value=EMPTY_FLOAT,
            binning=(n_bins_phi_cp, 0, 2*np.pi),
            x_title=rf"$\varphi_{{CP}} [{title_str}], \alpha < \pi/4$ (rad)",
        )
        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg2",
            expression=f"phi_cp_{the_ch}_reg2",
            null_value=EMPTY_FLOAT,
            binning=(n_bins_phi_cp, 0, 2*np.pi),
            x_title=rf"$\varphi_{{CP}} [{title_str}], \alpha \geq \pi/4$ (rad)",
        )
        # 2-bin histograms
        cfg.add_variable(
            name=f"phi_cp_{the_ch}_2bin",
            expression=f"phi_cp_{the_ch}_2bin",
            null_value=EMPTY_FLOAT,
            binning=(2, 0, 2*np.pi), 
            x_title=rf"$\varphi_{{CP}} [{title_str}]$ (rad)",
        )
        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg1_2bin",
            expression=f"phi_cp_{the_ch}_reg1_2bin",
            null_value=EMPTY_FLOAT,
            binning=(2, 0, 2*np.pi),
            x_title=rf"$\varphi_{{CP}} [{title_str}], \alpha < \pi/4$ (rad)",
        )
        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg2_2bin",
            expression=f"phi_cp_{the_ch}_reg2_2bin",
            null_value=EMPTY_FLOAT,
            binning=(2, 0, 2*np.pi),
            x_title=rf"$\varphi_{{CP}} [{title_str}], \alpha \geq \pi/4$ (rad)",
        )
        cfg.add_variable(
            name=f"alpha_{the_ch}",
            expression=f"alpha_{the_ch}",
            null_value=EMPTY_FLOAT,
            binning=(6, 0, np.pi/2),
            x_title=rf"$ \alpha [{title_str}] $(rad)",
        )

def add_dilepton_features(cfg: od.Config) -> None:
    channels = cfg.channels.names()
    ch_objects = DotDict.wrap({
        'etau'  : {'lep0':'Electron',
                   'lep1':'Tau'    },
        'mutau' : {'lep0':'Muon'    ,
                   'lep1':'Tau'    },
        'emu'   : {'lep0':'Electron',
                   'lep1':'Muon'   },
        'tautau': {'lep0':'Tau'     ,
                   'lep1':'Tau'    },
    })
    bin_split_factor = 4 #Used to define histograms for kinematic variables with finer binning
    for ch_str in channels:
        cfg.add_variable(
                name=f"{ch_str}_mvis",
                expression=f"hcand_{ch_str}.mass",
                null_value=EMPTY_FLOAT,
                binning=(25, 0.0, 250.0),
                unit="GeV",
                x_title=r"$m_{vis}$",
            )
        if ch_str in ['etau', 'mutau']:
            cfg.add_variable(
                name=f"{ch_str}_mt",
                expression=f"hcand_{ch_str}.mt",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 200.0),
                unit="GeV",
                x_title="$\\m_{T}$",
            )
        if ch_str == 'emu':
            cfg.add_variable(
                name=f"{ch_str}_mt_e",
                expression=f"hcand_{ch_str}.mt_e",
                null_value=EMPTY_FLOAT,
                binning=(25, 0.0, 250.0),
                unit="GeV",
                x_title="$m_{T}^{e}$",
            )
            cfg.add_variable(
                name=f"{ch_str}_mt_mu",
                expression=f"hcand_{ch_str}.mt_mu",
                null_value=EMPTY_FLOAT,
                binning=(25, 0.0, 250.0),
                unit="GeV",
                x_title="$m_{T}^{\\mu}$",
            )
            cfg.add_variable(
                name=f"{ch_str}_mt_emu",
                expression=f"hcand_{ch_str}.mt_emu",
                null_value=EMPTY_FLOAT,
                binning=(25, 0.0, 250.0),
                unit="GeV",
                x_title="$m_{T}^{e\\mu}$",
            )
            cfg.add_variable(
                name=f"{ch_str}_mt_tot",
                expression=f"hcand_{ch_str}.mt_tot",
                null_value=EMPTY_FLOAT,
                binning=(20, 0.0, 400.0),
                unit="GeV",
                x_title="$m_{T}^{TOT}$",
            )
        cfg.add_variable(
            name=f"{ch_str}_delta_r",
            expression=f"hcand_{ch_str}.delta_r",
            null_value=EMPTY_FLOAT,
            binning=(25, 0.2, 5.2),
            x_title=r"$\Delta R(\ell,\ell)$",
        )
        cfg.add_variable(
                name=f"{ch_str}_pt",
                expression=f"hcand_{ch_str}.pt",
                null_value=EMPTY_FLOAT,
                binning=(20, 0.0, 200.0),
                unit="GeV/c",
                x_title=r"$p_{T}(\ell\ell)$",
        )
        
        for lep in ['lep0','lep1']:
            if ch_str != 'tautau': lep_str = ch_objects[ch_str][lep].lower()
            else: lep_str = f'tau {lep[3:]}'
            cfg.add_variable(
                name=f"{ch_str}_{lep}_pt",
                expression=f"hcand_{ch_str}.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(20, 15, 215),
                unit="GeV",
                x_title= rf"{lep_str} $p_{{T}}$",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_eta",
                expression=f"hcand_{ch_str}.{lep}.eta",
                null_value=EMPTY_FLOAT,
                binning=(15, -2.5, 2.5),
                x_title=rf"{lep_str} $\eta$",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_phi",
                expression=f"hcand_{ch_str}.{lep}.phi",
                null_value=EMPTY_FLOAT,
                binning=(16, -3.3, 3.3),
                x_title=rf"{lep_str} $\phi$",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_mass",
                expression=f"hcand_{ch_str}.{lep}.mass",
                null_value=EMPTY_FLOAT,
                binning=(15, 0, 3),
                unit="GeV",
                x_title=f"{lep_str} mass",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_decayModePNet",
                expression=f"hcand_{ch_str}.{lep}.decayModePNet",
                null_value=EMPTY_FLOAT,
                binning=(12,0,12),
                unit="",
                x_title=rf"{lep_str} PNet decay mode",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_decayMode",
                expression=f"hcand_{ch_str}.{lep}.decayMode",
                null_value=EMPTY_FLOAT,
                binning=(12,0,12),
                unit="",
                x_title=rf"{lep_str} HPS decay mode",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_ip_sig",
                expression=f"hcand_{ch_str}.{lep}.ip_sig",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 10),
                unit="",
                x_title= rf"{lep_str} $\frac{{|IP|}}{{\sigma(IP)}}$",
            )
            for proj in ['x','y','z']:
                cfg.add_variable(
                    name=f"{ch_str}{lep}_ip_{proj}",
                    expression=f"hcand_{ch_str}.{lep}.IP{proj}",
                    null_value=EMPTY_FLOAT,
                    binning=(30, -0.002, 0.002),
                    unit="",
                    x_title= rf"{lep_str} $IP_{proj}$",
                )
            #Variables with finer binning
            cfg.add_variable(
                name=f"{ch_str}_{lep}_pt_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(30*bin_split_factor, 20, 80.),
                unit="GeV",
                x_title= rf"{lep_str} $p_{{T}}$",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_eta_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.eta",
                null_value=EMPTY_FLOAT,
                binning=(32*bin_split_factor, -3.2, 3.2),
                x_title=rf"{lep_str} $\eta$",
            )
            cfg.add_variable(
                name=f"{ch_str}_{lep}_phi_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.phi",
                null_value=EMPTY_FLOAT,
                binning=(32*bin_split_factor, -3.2, 3.2),
                x_title=rf"{lep_str} $\phi$",
            )
            ## FastMTT variables

            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_px",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.px",
                null_value=EMPTY_FLOAT,
                binning=(42, -10., 200.),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{x}^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_py",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.py",
                null_value=EMPTY_FLOAT,
                binning=(42, -10., 200.),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{y}^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_pz",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.pz",
                null_value=EMPTY_FLOAT,
                binning=(42, -10., 200.),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{z}^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_pt",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 200.),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{T}^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_eta",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.eta",
                null_value=EMPTY_FLOAT,
                binning=(25, -3.0, 3.0),
                unit="GeV",
                x_title=f"{lep} " + r"$\eta^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_phi",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.phi",
                null_value=EMPTY_FLOAT,
                binning=(32, -3.2, 3.2),
                unit="GeV",
                x_title=f"{lep} " + r"$\phi^{fastMTT}$",
            )
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_mass",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.mass",
                null_value=EMPTY_FLOAT,
                binning=(50, 0.01, 3.0),
                unit="GeV",
                x_title=f"{lep} " + r"$m^{fastMTT}$",
            )
        cfg.add_variable(
            name=f"hcand_{ch_str}_fastMTT_mass",
            expression=f"hcand_{ch_str}.fastMTT.mass",
            null_value=EMPTY_FLOAT,
            binning=(25, 0.0, 500.0),
            unit="GeV",
            x_title=r"$mass^{fastMTT}$",
        )
        
# =============================================================================
# MSSM BDT output variables
# =============================================================================

# Outputs produced directly by the new 10-feature 4-class BDT score producer:
#
#   bdt_raw_score_{ggphi,bbphi,dy,tt}_M{mass}
#   bdt_D_sig_M{mass}
#   bdt_D_ggphi_M{mass}
#   bdt_D_bbphi_M{mass}
#   bdt_Disc_ggphi_M{mass}
#   bdt_Disc_bbphi_M{mass}
#   bdt_D_DY_M{mass}
#   bdt_D_TT_M{mass}
#   bdt_D_bbphi_sig_M{mass}
#   bdt_D_ggphi_sig_M{mass}
#   bdt_cat_M{mass}
#
# The 2D flattened variables are still assumed to be produced by the dedicated
# BDT 2D-variable producer. Their binning is derived from the same adaptive edge
# JSON files written by the training post-processing.

BDT_CLASS_LABELS = ("ggphi", "bbphi", "dy", "tt")

BDT_CLASS_TITLES = {
    "ggphi": r"gg$\phi$($\phi\rightarrow\tau\tau$)",
    "bbphi": r"bb$\phi$($\phi\rightarrow\tau\tau$)",
    "dy": "DY",
    "tt": r"t$\bar{t}$",
}

# Discriminants with their own adaptive binning in the new training script.
BDT_1D_DISCRIMINANTS = {
    "D_sig": {
        "title": r"$D_{\mathrm{sig}}$",
        "binning_source": "D_sig",
    },
    "D_ggphi": {
        "title": r"$D_{\mathrm{gg}\phi}$",
        "binning_source": "D_ggphi",
    },
    "D_bbphi": {
        "title": r"$D_{\mathrm{bb}\phi}$",
        "binning_source": "D_bbphi",
    },
    "Disc_ggphi": {
        "title": (
            r"$D_{\mathrm{gg}\phi}/"
            r"(D_{\mathrm{gg}\phi}+D_{\mathrm{bb}\phi})$"
        ),
        "binning_source": "Disc_ggphi",
    },
    "Disc_bbphi": {
        "title": (
            r"$D_{\mathrm{bb}\phi}/"
            r"(D_{\mathrm{gg}\phi}+D_{\mathrm{bb}\phi})$"
        ),
        "binning_source": "Disc_bbphi",
    },
    "D_DY": {
        "title": r"$D_{\mathrm{DY}}$",
        "binning_source": "D_DY",
    },
    "D_TT": {
        "title": r"$D_{\mathrm{TT}}$",
        "binning_source": "D_TT",
    },
}

# Diagnostic signal-splitting discriminants produced by the new score producer.
# They are numerically equivalent to the Disc_* variables, so reuse those
# adaptive edges when no dedicated edge file exists.
BDT_DIAGNOSTIC_1D_DISCRIMINANTS = {
    "D_ggphi_sig": {
        "title": r"$P_{\mathrm{gg}\phi}/(P_{\mathrm{gg}\phi}+P_{\mathrm{bb}\phi})$",
        "binning_source": "Disc_ggphi",
    },
    "D_bbphi_sig": {
        "title": r"$P_{\mathrm{bb}\phi}/(P_{\mathrm{gg}\phi}+P_{\mathrm{bb}\phi})$",
        "binning_source": "Disc_bbphi",
    },
}

# Backward-compatible aliases for older plot/datacard configs.
BDT_DISCRIMINANT_ALIASES = {
    "D_dy": "D_DY",
    "D_tt": "D_TT",
}

# Flattened 2D variables produced by MSSM_H_tt/production/bdt_2d_variables.py.
#
# Rectangular 2D binning uses:
#
#   flat_index = ix * n_y_bins + iy
#   n_flat_bins = n_x_bins * n_y_bins
#
# For y-merged 2D binnings, the JSON contains a different list of y edges in
# each x bin.  In that case the flattened axis must contain one bin per
# irregular 2D cell:
#
#   n_flat_bins = sum(n_y_bins_in_this_x_bin for each x bin)
#
# The actual producer must use the same cumulative-offset convention when
# assigning flat indices.
BDT_2D_FLATTENED_PAIRS = (
    ("D_sig_vs_Disc_ggphi", "D_sig", "Disc_ggphi"),
    ("D_sig_vs_Disc_bbphi", "D_sig", "Disc_bbphi"),
    ("D_sig_vs_D_ggphi", "D_sig", "D_ggphi"),
    ("D_sig_vs_D_bbphi", "D_sig", "D_bbphi"),
    ("D_ggphi_vs_D_bbphi", "D_ggphi", "D_bbphi"),
)

# These pairs have a dedicated y-merged 2D binning JSON in the training
# output.  The file names follow the training script convention
#   2D_{y_name}_vs_{x_name}_yMergedMinWeightedBkg_edges_M{mass}_{tag}.json
BDT_2D_YMERGED_PAIRS = {
    ("D_sig", "Disc_ggphi"),
    ("D_sig", "Disc_bbphi"),
}


def _bdt_all_1d_discriminants() -> dict:
    out = {}
    out.update(BDT_1D_DISCRIMINANTS)
    out.update(BDT_DIAGNOSTIC_1D_DISCRIMINANTS)
    return out


def _bdt_discriminant_binning_source(discriminant: str) -> str:
    """
    Return the discriminant whose adaptive binning should be used.

    In the new training, Disc_ggphi and Disc_bbphi have their own adaptive
    edge files. The diagnostic D_*_sig variables reuse those Disc_* edges.
    """
    all_discriminants = _bdt_all_1d_discriminants()

    if discriminant in all_discriminants:
        return all_discriminants[discriminant]["binning_source"]

    if discriminant in BDT_DISCRIMINANT_ALIASES:
        return BDT_DISCRIMINANT_ALIASES[discriminant]

    return discriminant


def _bdt_n_bins_from_binning(binning) -> int:
    """
    Return the number of bins from either:
      - regular binning: (n_bins, x_min, x_max)
      - variable binning: [edge0, edge1, ...]
    """
    b = list(binning)

    if (
        len(b) == 3
        and isinstance(b[0], (int, np.integer))
        and b[0] > 0
        and float(b[1]) < float(b[2])
    ):
        return int(b[0])

    return len(b) - 1


def _bdt_binning(mass: int, discriminant: str):
    """
    Return the adaptive binning for a BDT output variable.
    """
    source = _bdt_discriminant_binning_source(discriminant)
    return _read_bdt_adaptive_binning(mass, source)



def _bdt_ymerged_2d_edges_path(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
) -> Path:
    """
    Return the y-merged 2D edge JSON path for one pair.

    This matches the training output naming, for example
      2D_Disc_ggphi_vs_D_sig_yMergedMinWeightedBkg_edges_M100_combined_crossApplied.json
    for x=D_sig, y=Disc_ggphi.
    """
    mass = int(mass)
    safe_pair = f"{y_discriminant}_vs_{x_discriminant}".replace("/", "_")
    return (
        BDT_OUTPUT_BASE
        / f"M{mass}"
        / "adaptive_discriminant_rebinning"
        / f"M{mass}"
        / "two_dimensional_discriminants"
        / BDT_2D_ADAPTIVE_SUBDIR
        / f"2D_{safe_pair}_{BDT_2D_YMERGED_TOKEN}_edges_M{mass}_{BDT_ADAPTIVE_TAG}.json"
    )


def _bdt_extract_ymerged_2d_binning_from_json(data) -> dict:
    """
    Extract the irregular 2D binning from a y-merged 2D JSON.

    Expected training JSON structure:
      {
        "x_edges": [...],
        "y_edges_by_x_bin": [
          {"x_bin": 1, "x_low": ..., "x_high": ..., "y_edges": [...]},
          ...
        ],
        "n_cells": ...
      }

    The flattened-axis convention is cumulative in x:
      flat_index = x_bin_offset[ix] + iy

    where
      x_bin_offset[ix] = sum(n_y_bins in previous x bins).
    """
    if not isinstance(data, dict):
        raise TypeError("2D adaptive binning JSON payload is not a dictionary")

    if "x_edges" not in data:
        raise KeyError("Could not find 'x_edges' in 2D JSON payload")
    if "y_edges_by_x_bin" not in data:
        raise KeyError("Could not find 'y_edges_by_x_bin' in 2D JSON payload")

    x_edges = [float(x) for x in data["x_edges"]]
    if len(x_edges) < 2:
        raise ValueError("Invalid 2D binning: x_edges has fewer than two entries")

    y_entries = data["y_edges_by_x_bin"]
    if not isinstance(y_entries, list):
        raise TypeError("Invalid 2D binning: y_edges_by_x_bin is not a list")
    if len(y_entries) != len(x_edges) - 1:
        raise ValueError(
            "Invalid 2D binning: len(y_edges_by_x_bin) does not match "
            "len(x_edges)-1"
        )

    y_edges_by_x_bin = []
    n_y_bins_by_x_bin = []
    x_bin_offsets = []
    offset = 0

    for ix, entry in enumerate(y_entries):
        if not isinstance(entry, dict):
            raise TypeError(f"Invalid 2D binning: y entry {ix} is not a dictionary")

        y_edges = [float(y) for y in entry.get("y_edges", [])]
        if len(y_edges) < 2:
            raise ValueError(f"Invalid 2D binning: x bin {ix + 1} has fewer than two y edges")

        n_y_bins = len(y_edges) - 1
        x_bin_offsets.append(int(offset))
        y_edges_by_x_bin.append(y_edges)
        n_y_bins_by_x_bin.append(int(n_y_bins))
        offset += n_y_bins

    n_cells_from_edges = int(offset)
    n_cells_from_json = data.get("n_cells")

    if n_cells_from_json is not None and int(n_cells_from_json) != n_cells_from_edges:
        raise ValueError(
            "Invalid 2D binning: n_cells in JSON does not match the number "
            "computed from y_edges_by_x_bin "
            f"({int(n_cells_from_json)} != {n_cells_from_edges})"
        )

    if n_cells_from_edges <= 0:
        raise ValueError("Invalid y-merged 2D binning: extracted zero flattened bins")

    return {
        "x_edges": x_edges,
        "y_edges_by_x_bin": y_edges_by_x_bin,
        "n_y_bins_by_x_bin": n_y_bins_by_x_bin,
        "x_bin_offsets": x_bin_offsets,
        "n_cells": n_cells_from_edges,
    }


def _bdt_extract_n_flat_bins_from_ymerged_2d_json(data) -> int:
    """
    Extract the number of flattened bins from a y-merged 2D JSON.
    """
    return int(_bdt_extract_ymerged_2d_binning_from_json(data)["n_cells"])


def _read_bdt_ymerged_2d_flattened_binning(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
):
    """
    Return flattened binning from the y-merged 2D JSON when available.

    The returned binning is regular in the flattened coordinate:
      (n_irregular_2d_cells, 0, n_irregular_2d_cells)

    Return None when the pair is not configured for y-merging or when the JSON
    is not available/readable, so the caller can fall back to rectangular
    n_x_bins * n_y_bins binning.
    """
    if (str(x_discriminant), str(y_discriminant)) not in BDT_2D_YMERGED_PAIRS:
        return None

    path = _bdt_ymerged_2d_edges_path(mass, x_discriminant, y_discriminant)
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text())
        n_flat_bins = _bdt_extract_n_flat_bins_from_ymerged_2d_json(data)
        return (n_flat_bins, 0, n_flat_bins)
    except Exception:
        return None

def _bdt_flattened_2d_binning(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
):
    """
    Return the flattened 2D binning for a pair of discriminants.

    Priority:
      1. y-merged 2D JSON for pairs with dedicated irregular 2D binning
      2. rectangular fallback from the 1D adaptive edges

    The axis is always the flattened coordinate.  For y-merged binning, the
    number of bins is the number of irregular 2D cells, not n_x * n_y from
    the independent 1D binnings.
    """
    ymerged_binning = _read_bdt_ymerged_2d_flattened_binning(
        mass,
        x_discriminant,
        y_discriminant,
    )
    if ymerged_binning is not None:
        return ymerged_binning

    x_binning = _bdt_binning(mass, x_discriminant)
    y_binning = _bdt_binning(mass, y_discriminant)

    n_x_bins = _bdt_n_bins_from_binning(x_binning)
    n_y_bins = _bdt_n_bins_from_binning(y_binning)

    n_flat_bins = n_x_bins * n_y_bins

    return (n_flat_bins, 0, n_flat_bins)


def _bdt_discriminant_title(discriminant: str) -> str:
    """
    Return a readable title for standard, Disc_* and diagnostic discriminants.
    """
    all_discriminants = _bdt_all_1d_discriminants()

    if discriminant in all_discriminants:
        return all_discriminants[discriminant]["title"]

    if discriminant in BDT_DISCRIMINANT_ALIASES:
        target = BDT_DISCRIMINANT_ALIASES[discriminant]
        return all_discriminants[target]["title"]

    return str(discriminant)


def add_mssm_bdt_output(cfg: od.Config) -> None:
    """
    Register the per-mass outputs of the current MSSM e-mu 10-feature,
    four-class BDT producer.

    Feature/training convention:
      0 -> ggphi_phitautau
      1 -> bbphi_phitautau
      2 -> DY
      3 -> TT

    Four-region convention:
      bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)

    Main region-specific fit variables:
      ggphi region -> bdt_D_ggphi_M{mass}
      bbphi region -> bdt_D_bbphi_M{mass}
      DY     region -> bdt_D_DY_M{mass}
      TT     region -> bdt_D_TT_M{mass}

    Additional outputs from the new score producer:
      bdt_Disc_ggphi_M{mass}
      bdt_Disc_bbphi_M{mass}
      bdt_D_ggphi_sig_M{mass}
      bdt_D_bbphi_sig_M{mass}
    """
    from MSSM_H_tt.config.mass_points import read_bdt_masses
    MASS_POINTS = read_bdt_masses()

    for m in MASS_POINTS:
        # ------------------------------------------------------------------
        # Raw four-class probabilities from the BDT producer
        # ------------------------------------------------------------------
        for label in BDT_CLASS_LABELS:
            cfg.add_variable(
                name=f"bdt_raw_score_{label}_M{m}",
                expression=f"bdt_raw_score_{label}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=BDT_DEFAULT_SCORE_BINNING,
                x_title=f"BDT probability for {BDT_CLASS_TITLES[label]} (M={m} GeV)",
            )

        # ------------------------------------------------------------------
        # Standard and new 1D BDT discriminants
        # ------------------------------------------------------------------
        for discr_name, discr_info in _bdt_all_1d_discriminants().items():
            cfg.add_variable(
                name=f"bdt_{discr_name}_M{m}",
                expression=f"bdt_{discr_name}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_bdt_binning(m, discr_name),
                x_title=f"{discr_info['title']} (M={m} GeV)",
            )

        # ------------------------------------------------------------------
        # Backward-compatible aliases for older plot configs that used
        # bdt_D_dy_M{m} and bdt_D_tt_M{m}.
        # ------------------------------------------------------------------
        for alias, target in BDT_DISCRIMINANT_ALIASES.items():
            cfg.add_variable(
                name=f"bdt_{alias}_M{m}",
                expression=f"bdt_{target}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_bdt_binning(m, target),
                x_title=f"{_bdt_discriminant_title(target)} (M={m} GeV)",
            )

        # ------------------------------------------------------------------
        # Flattened 2D BDT variables
        # ------------------------------------------------------------------
        for pair_name, x_discr, y_discr in BDT_2D_FLATTENED_PAIRS:
            cfg.add_variable(
                name=f"bdt_{pair_name}_M{m}",
                expression=f"bdt_{pair_name}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_bdt_flattened_2d_binning(m, x_discr, y_discr),
                x_title=(
                    "Flattened 2D bin: "
                    f"{_bdt_discriminant_title(x_discr)} vs "
                    f"{_bdt_discriminant_title(y_discr)} "
                    f"(M={m} GeV)"
                ),
            )

        # ------------------------------------------------------------------
        # Four BDT regions
        # ------------------------------------------------------------------
        cfg.add_variable(
            name=f"bdt_cat_M{m}",
            expression=f"bdt_cat_M{m}",
            binning=(4, -0.5, 3.5),
            discrete_x=True,
            x_title=(
                r"BDT category: "
                r"0=gg$\phi$, 1=bb$\phi$, 2=DY, 3=t$\bar{t}$ "
                f"(M={m} GeV)"
            ),
        )

def add_emu_phi_cp_features(cfg: od.Config) -> None:
    cfg.add_variable(
        name="phi_cp_emu",
        expression="phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(16, 0.0, 2 * np.pi),
        x_title=r"$\varphi_{CP}^{e\mu}$ (rad)",
    )
    cfg.add_variable(
        name="cos_phi_cp_emu",
        expression="cos_phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(20, -1.0, 1.0),
        x_title=r"$\cos(\varphi_{CP}^{e\mu})$",
    )

    cfg.add_variable(
        name="sin_phi_cp_emu",
        expression="sin_phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(20, -1.0, 1.0),
        x_title=r"$\sin(\varphi_{CP}^{e\mu})$",
    )    

def add_variables(cfg: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    add_common_features(cfg)
    add_lepton_features(cfg)
    add_jet_features(cfg)
    add_highlevel_features(cfg)
    add_weight_features(cfg)
    add_cutflow_features(cfg)
    add_dilepton_features(cfg)
    add_mssm_bdt_output(cfg)
    add_emu_phi_cp_features(cfg)