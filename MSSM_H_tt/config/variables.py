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
BDT_OUTPUT_BASE = Path(
    "/eos/project/d/desytau/public/jmalvaso/"
    "bdt_4_classes_phi_no_DY_tail_focus_normWeightTraining_clippedJetCounts"
)
BDT_ADAPTIVE_TAG = "combined_crossApplied"
BDT_DEFAULT_SCORE_BINNING = (30, 0.0, 1.0)


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

# Derived 1D BDT variables produced by MSSM_H_tt/production/bdt_2d_variables.py
#
#   bdt_Disc_ggphi_M{mass}
#   bdt_Disc_bbphi_M{mass}
#
# Binning convention:
#   Disc_ggphi uses the same adaptive binning as D_ggphi
#   Disc_bbphi uses the same adaptive binning as D_bbphi
#
BDT_DERIVED_1D_DISCRIMINANTS = {
    "Disc_ggphi": {
        "source": "D_ggphi",
        "title": (
            r"$D_{\mathrm{gg}\phi}/"
            r"(D_{\mathrm{gg}\phi}+D_{\mathrm{bb}\phi})$"
        ),
    },
    "Disc_bbphi": {
        "source": "D_bbphi",
        "title": (
            r"$D_{\mathrm{bb}\phi}/"
            r"(D_{\mathrm{gg}\phi}+D_{\mathrm{bb}\phi})$"
        ),
    },
}


# Flattened 2D variables produced by MSSM_H_tt/production/bdt_2d_variables.py.
#
# The producer stores a flattened bin coordinate:
#
#   flat_index = ix * n_y_bins + iy
#
# Therefore the plotting variable must use:
#
#   binning = (n_x_bins * n_y_bins, 0, n_x_bins * n_y_bins)
#
BDT_2D_FLATTENED_PAIRS = (
    ("D_sig_vs_D_ggphi", "D_sig", "D_ggphi"),
    ("D_sig_vs_D_bbphi", "D_sig", "D_bbphi"),
    ("D_ggphi_vs_D_bbphi", "D_ggphi", "D_bbphi"),

    # New derived-disc flattened 2D variables
    ("D_sig_vs_Disc_ggphi", "D_sig", "Disc_ggphi"),
    ("D_sig_vs_Disc_bbphi", "D_sig", "Disc_bbphi"),
)


def _bdt_discriminant_binning_source(discriminant: str) -> str:
    """
    Return the discriminant whose adaptive binning should be used.

    For standard variables:
      D_sig -> D_sig
      D_ggphi -> D_ggphi
      D_bbphi -> D_bbphi

    For derived variables:
      Disc_ggphi -> D_ggphi
      Disc_bbphi -> D_bbphi
    """
    if discriminant in BDT_DERIVED_1D_DISCRIMINANTS:
        return BDT_DERIVED_1D_DISCRIMINANTS[discriminant]["source"]

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


def _bdt_flattened_2d_binning(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
):
    """
    Return the flattened 2D binning for a pair of discriminants.

    The producer stores values in:
      [0, n_x_bins * n_y_bins)

    with bin centers:
      flat_index + 0.5
    """
    x_source = _bdt_discriminant_binning_source(x_discriminant)
    y_source = _bdt_discriminant_binning_source(y_discriminant)

    x_binning = _read_bdt_adaptive_binning(mass, x_source)
    y_binning = _read_bdt_adaptive_binning(mass, y_source)

    n_x_bins = _bdt_n_bins_from_binning(x_binning)
    n_y_bins = _bdt_n_bins_from_binning(y_binning)

    n_flat_bins = n_x_bins * n_y_bins

    return (n_flat_bins, 0, n_flat_bins)


def _bdt_discriminant_title(discriminant: str, discriminants: dict) -> str:
    """
    Return a readable title for both standard and derived discriminants.
    """
    if discriminant in BDT_DERIVED_1D_DISCRIMINANTS:
        return BDT_DERIVED_1D_DISCRIMINANTS[discriminant]["title"]

    return discriminants[discriminant]


def add_mssm_bdt_output(cfg: od.Config) -> None:
    """
    Register the per-mass outputs of the current MSSM e-mu 4-class BDT producer.

    Four-region convention:
      bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)

    Class convention:
      0 -> ggphi
      1 -> bbphi
      2 -> DY
      3 -> TT

    Region-specific fit variables:
      ggphi region -> bdt_D_ggphi_M{mass}
      bbphi region -> bdt_D_bbphi_M{mass}
      DY     region -> bdt_D_DY_M{mass}
      TT     region -> bdt_D_TT_M{mass}

    Additional derived 1D variables:
      bdt_Disc_ggphi_M{mass}
      bdt_Disc_bbphi_M{mass}

    Flattened 2D variables:
      bdt_D_sig_vs_D_ggphi_M{mass}
      bdt_D_sig_vs_D_bbphi_M{mass}
      bdt_D_ggphi_vs_D_bbphi_M{mass}
      bdt_D_sig_vs_Disc_ggphi_M{mass}
      bdt_D_sig_vs_Disc_bbphi_M{mass}
    """
    from MSSM_H_tt.config.mass_points import read_bdt_masses
    MASS_POINTS = read_bdt_masses()

    class_labels = ["ggphi", "bbphi", "dy", "tt"]

    class_titles = {
        "ggphi": r"gg$\phi$($\phi\rightarrow\tau\tau$)",
        "bbphi": r"bb$\phi$($\phi\rightarrow\tau\tau$)",
        "dy": "DY",
        "tt": r"t$\bar{t}$",
    }

    discriminants = {
        "D_sig": r"$D_{\mathrm{sig}}$",
        "D_ggphi": r"$D_{\mathrm{gg}\phi}$",
        "D_bbphi": r"$D_{\mathrm{bb}\phi}$",
        "D_DY": r"$D_{\mathrm{DY}}$",
        "D_TT": r"$D_{\mathrm{TT}}$",
    }

    # Optional plotting aliases with lower-case background names.
    # Keep them only if downstream plotting/config code still expects D_dy/D_tt.
    discriminant_aliases = {
        "D_dy": "D_DY",
        "D_tt": "D_TT",
    }

    for m in MASS_POINTS:
        # ---------------------------------------------------------------------
        # Raw four-class probabilities from the BDT producer
        # ---------------------------------------------------------------------
        for label in class_labels:
            cfg.add_variable(
                name=f"bdt_raw_score_{label}_M{m}",
                expression=f"bdt_raw_score_{label}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=(30, 0.0, 1.0),
                x_title=f"BDT probability for {class_titles[label]} (M={m} GeV)",
            )

        # ---------------------------------------------------------------------
        # Standard 1D BDT discriminants
        # ---------------------------------------------------------------------
        for discr_name, discr_title in discriminants.items():
            cfg.add_variable(
                name=f"bdt_{discr_name}_M{m}",
                expression=f"bdt_{discr_name}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_read_bdt_adaptive_binning(m, discr_name),
                x_title=f"{discr_title} (M={m} GeV)",
            )

        # ---------------------------------------------------------------------
        # New derived 1D BDT discriminants
        #
        # Produced columns:
        #   bdt_Disc_ggphi_M{m}
        #   bdt_Disc_bbphi_M{m}
        # ---------------------------------------------------------------------
        for discr_name, discr_info in BDT_DERIVED_1D_DISCRIMINANTS.items():
            source_discr = discr_info["source"]
            discr_title = discr_info["title"]

            cfg.add_variable(
                name=f"bdt_{discr_name}_M{m}",
                expression=f"bdt_{discr_name}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_read_bdt_adaptive_binning(m, source_discr),
                x_title=f"{discr_title} (M={m} GeV)",
            )

        # ---------------------------------------------------------------------
        # Backward-compatible aliases for older plot configs that used
        # bdt_D_dy_M{m} and bdt_D_tt_M{m}.
        # ---------------------------------------------------------------------
        for alias, target in discriminant_aliases.items():
            cfg.add_variable(
                name=f"bdt_{alias}_M{m}",
                expression=f"bdt_{target}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_read_bdt_adaptive_binning(m, target),
                x_title=f"{discriminants[target]} (M={m} GeV)",
            )

        # ---------------------------------------------------------------------
        # Flattened 2D BDT variables
        #
        # Producer output convention:
        #   flat_index = ix * n_y_bins + iy
        #   stored value = flat_index + 0.5
        #
        # Config binning:
        #   (n_x_bins * n_y_bins, 0, n_x_bins * n_y_bins)
        # ---------------------------------------------------------------------
        for pair_name, x_discr, y_discr in BDT_2D_FLATTENED_PAIRS:
            x_title = _bdt_discriminant_title(x_discr, discriminants)
            y_title = _bdt_discriminant_title(y_discr, discriminants)

            cfg.add_variable(
                name=f"bdt_{pair_name}_M{m}",
                expression=f"bdt_{pair_name}_M{m}",
                null_value=EMPTY_FLOAT,
                binning=_bdt_flattened_2d_binning(m, x_discr, y_discr),
                x_title=(
                    f"Flattened 2D bin: {x_title} vs {y_title} "
                    f"(M={m} GeV)"
                ),
            )

        # ---------------------------------------------------------------------
        # Four BDT regions
        # ---------------------------------------------------------------------
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