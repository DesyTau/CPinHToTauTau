### This config is used for listing the variables used in the analysis ###

from columnflow.config_util import add_category

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

from columnflow.util import maybe_import

np = maybe_import("numpy")

import os
import json
from pathlib import Path


# =============================================================================
# BDT adaptive binning configuration
# =============================================================================

# All BDT binning definitions should come from the same training /
# post-processing output.
#
# Preferred environment variable:
#
#   MSSM_BDT_BINNING_BASE
#
# For backward compatibility, MSSM_BDT_2D_BINNING_BASE is also accepted.
#
# When neither variable is set, use the existing CERN EOS location.
BDT_DEFAULT_OUTPUT_BASE = (
    "/eos/project/d/desytau/public/jmalvaso/"
    "bdt_3_classes_10_features_clippedJetCounts"
)

BDT_OUTPUT_BASE = Path(
    os.environ.get(
        "MSSM_BDT_BINNING_BASE",
        os.environ.get(
            "MSSM_BDT_2D_BINNING_BASE",
            BDT_DEFAULT_OUTPUT_BASE,
        ),
    )
)

BDT_ADAPTIVE_TAG = "combined_crossApplied"

# This fallback is kept for ordinary 1D diagnostic variables.
# It is NEVER used for the optimized y-merged 2D datacard variables.
BDT_DEFAULT_SCORE_BINNING = (30, 0.0, 1.0)

# Location / naming of the irregular y-merged 2D binning JSON files produced by
# adaptive_plot_2d_discriminant_pair(...).
BDT_2D_ADAPTIVE_SUBDIR = "independent_adaptive"
BDT_2D_YMERGED_TOKEN = "yMergedMinWeightedBkg"


def _bdt_mass_region_name(mass: int) -> str:
    mass = int(mass)

    if mass <= 250:
        return "lowMass_Mle250"

    return "highMass_Mgt250"


def _bdt_common_edges_path(
    mass: int,
    discriminant: str,
) -> Path:
    region_name = _bdt_mass_region_name(mass)

    return (
        BDT_OUTPUT_BASE
        / "adaptive_discriminant_rebinning"
        / f"common_{region_name}"
        / discriminant
        / (
            f"{discriminant}_common_region_edges_"
            f"{region_name}_{BDT_ADAPTIVE_TAG}.json"
        )
    )


def _bdt_per_mass_edges_path(
    mass: int,
    discriminant: str,
) -> Path:
    mass = int(mass)

    return (
        BDT_OUTPUT_BASE
        / f"M{mass}"
        / "adaptive_discriminant_rebinning"
        / f"M{mass}"
        / discriminant
        / (
            f"{discriminant}_adaptive_edges_"
            f"M{mass}_{BDT_ADAPTIVE_TAG}.json"
        )
    )


def _extract_edges_from_bdt_json(data):
    """
    Extract 1D adaptive edges from a training JSON.

    The common-region JSON stores edges at top level:

        {
            "edges": [...]
        }

    while the per-mass JSON stores them inside:

        {
            "binning_summary": {
                "edges": [...]
            }
        }
    """
    if isinstance(data, dict):
        if "edges" in data:
            return data["edges"]

        if (
            "binning_summary" in data
            and "edges" in data["binning_summary"]
        ):
            return data["binning_summary"]["edges"]

    raise KeyError(
        "Could not find adaptive bin edges in JSON payload"
    )


def _read_bdt_adaptive_binning(
    mass: int,
    discriminant: str,
):
    """
    Return adaptive bin edges for an ordinary 1D BDT discriminant.

    Priority:

      1. common low/high-mass adaptive edges,
      2. per-mass adaptive edges,
      3. fixed 30-bin fallback.

    IMPORTANT:
    This fallback is only for ordinary 1D variables.

    Optimized y-merged 2D variables have their own strict reader below and
    are never permitted to use the 30 x 30 rectangular fallback.
    """
    paths = (
        _bdt_common_edges_path(
            mass,
            discriminant,
        ),
        _bdt_per_mass_edges_path(
            mass,
            discriminant,
        ),
    )

    for path in paths:
        if not path.is_file():
            continue

        try:
            data = json.loads(
                path.read_text()
            )

            edges = [
                float(x)
                for x in _extract_edges_from_bdt_json(data)
            ]

            if len(edges) >= 2:
                return edges

        except Exception:
            # Preserve the existing behavior for ordinary 1D diagnostic
            # variables.
            pass

    return BDT_DEFAULT_SCORE_BINNING


# =============================================================================
# Columns to retain
# =============================================================================

def keep_columns(cfg: od.Config) -> None:
    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # TauProds
            "TauProd.*",
            "GenPart.*",
            "GenZ.*",

            # general event info
            "run",
            "luminosityBlock",
            "event",
            "PV.npvs",
            "Pileup.nTrueInt",
            "Pileup.nPU",
            "genWeight",
            "LHEWeight.originalXWGTUP",
            "HTXS_njets*",
            "LHE_Njets",
            "LHEScaleWeight*",
            "PSWeight*",
            "weight",
            "zpt_weight",
            "muon_weight_nom",
            "mc_weight",
            "tau_weight_nom",
        } | {
            f"PuppiMET.{var}"
            for var in [
                "pt",
                "phi",
                "significance",
                "covXX",
                "covXY",
                "covYY",
                "ptUnclusteredUp",
                "ptUnclusteredDown",
                "phiUnclusteredUp",
                "phiUnclusteredDown",
            ]
        } | {
            f"MET.{var}"
            for var in [
                "pt",
                "phi",
                "significance",
                "covXX",
                "covXY",
                "covYY",
            ]
        } | {
            f"Jet.{var}"
            for var in [
                "pt",
                "eta",
                "phi",
                "mass",
                "jetId",
                "btagDeepFlavB",
                "hadronFlavour",
                "btagPNetB",
                "neEmEF",
                "chHEF",
                "neHEF",
                "chHEF",
                "muEF",
                "chEmEF",
                "neMultiplicity",
                "chMultiplicity",
            ]
        } | {
            f"Tau.{var}"
            for var in [
                "pt",
                "eta",
                "phi",
                "mass",
                "dxy",
                "dz",
                "charge",
                "rawDeepTau2018v2p5VSjet",
                "idDeepTau2018v2p5VSjet",
                "idDeepTau2018v2p5VSe",
                "idDeepTau2018v2p5VSmu",
                "decayMode",
                "decayModePNet",
                "genPartFlav",
                "rawIdx",
                "pt_no_tes",
                "mass_no_tes",
                "IPx",
                "IPy",
                "IPz",
                "ip_sig",
                "jetIdx",
            ]
        } | {
            f"Muon.{var}"
            for var in [
                "pt",
                "eta",
                "phi",
                "mass",
                "dxy",
                "dz",
                "charge",
                "decayMode",
                "pfRelIso04_all",
                "mT",
                "rawIdx",
                "IPx",
                "IPy",
                "IPz",
                "ip_sig",
                "jetIdx",
            ]
        } | {
            f"Electron.{var}"
            for var in [
                "pt",
                "eta",
                "phi",
                "mass",
                "dxy",
                "dz",
                "charge",
                "decayMode",
                "pfRelIso03_all",
                "mT",
                "rawIdx",
                "IPx",
                "IPy",
                "IPz",
                "ip_sig",
                "jetIdx",
                "pt_no_scaling_smearing",
            ]
        } | {
            f"{var}_triggerd"
            for var in [
                "single_electron",
                "cross_electron",
                "single_muon",
                "cross_muon",
                "cross_tau",
            ]
        } | {
            f"matched_triggerID_{var}"
            for var in [
                "e",
                "mu",
                "tau",
            ]
        } | {
            f"TrigObj.{var}"
            for var in [
                "id",
                "pt",
                "eta",
                "phi",
                "filterBits",
            ]
        } | {
            f"TauSpinner.weight_cp_{var}"
            for var in [
                "0",
                "0_alt",
                "0p25",
                "0p25_alt",
                "0p375",
                "0p375_alt",
                "0p5",
                "0p5_alt",
                "minus0p25",
                "minus0p25_alt",
            ]
        } | {
            f"hcand_emu.lep0.{var}"
            for var in [
                "jetIdx",
                "pt",
                "eta",
                "phi",
                "mass",
                "ip_sig",
                "charge",
            ]
        } | {
            f"hcand_emu.lep1.{var}"
            for var in [
                "jetIdx",
                "pt",
                "eta",
                "phi",
                "mass",
                "ip_sig",
                "charge",
                "pfRelIso04_all",
            ]
        } | {
            "GenTau.*",
            "GenTauProd.*",

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
            f"hcandprod.{var}"
            for var in [
                "pt",
                "eta",
                "phi",
                "mass",
                "charge",
                "pdgId",
                "tauIdx",
            ]
        } | {
            "hcand_emu.*",
            "tau_decay_prods*",
        } | {
            "is_b_vetoed",
            "channel_id",
        } | {
            ColumnCollection.ALL_FROM_SELECTOR
        },

        "cf.MergeSelectionMasks": {
            "normalization_weight",
            "cutflow.*",
            "process_id",
            "category_ids",
        } | {
            "bdt_*",
        },

        "cf.UniteColumns": {
            "*",
        },
    })


# =============================================================================
# Common features
# =============================================================================

def add_common_features(cfg: od.Config) -> None:
    """
    Adds common features.
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


# =============================================================================
# Lepton features
# =============================================================================

def add_lepton_features(cfg: od.Config) -> None:
    """
    Adds lepton features only.
    """
    cfg.add_variable(
        name="electron_1_pt_no_scaling_smearing",
        expression="Electron.pt_no_scaling_smearing[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r" Electron $p_{T}$ no scaling or smearing",
    )

    for obj in [
        "Electron",
        "Muon",
        "Tau",
    ]:
        for i in range(2):
            cfg.add_variable(
                name=f"{obj.lower()}_{i + 1}_pt",
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 200.0),
                unit="GeV",
                x_title=obj + r" $p_{T}$",
            )

            cfg.add_variable(
                name=f"{obj.lower()}_{i + 1}_phi",
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(32, -3.2, 3.2),
                x_title=obj + r" $\phi$",
            )

            cfg.add_variable(
                name=f"{obj.lower()}_{i + 1}_eta",
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(25, -2.5, 2.5),
                x_title=obj + r" $\eta$",
            )

        cfg.add_variable(
            name=f"{obj.lower()}_ip_sig",
            expression=f"{obj}.ip_sig",
            null_value=EMPTY_FLOAT,
            binning=(40, 0.0, 10.0),
            unit="",
            x_title=obj + r"$\frac{|IP|}{\sigma(IP)}$",
        )


# =============================================================================
# Jet features
# =============================================================================

def add_jet_features(cfg: od.Config) -> None:
    """
    Adds jet features only.
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
        name="lead_jet_is_btagged",
        expression="lead_jet_is_btagged",
        binning=(2, 0, 2),
        x_title="Leading jet is b-tagged",
        discrete_x=True,
    )

    cfg.add_variable(
        name="sublead_jet_is_btagged",
        expression="sublead_jet_is_btagged",
        binning=(2, 0, 2),
        x_title="Subleading jet is b-tagged",
        discrete_x=True,
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
        binning=(12, -6, 6),
        x_title="$\\Delta \\eta_{jj}$",
    )

    cfg.add_variable(
        name="dijet_delta_phi",
        expression="dijet.deltaphi",
        null_value=EMPTY_FLOAT,
        binning=(12, -6, 6),
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
        binning=(15, 0, 5),
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
        binning=(12, -6, 6),
        x_title="$\\Delta \\eta_{bb}$",
    )

    cfg.add_variable(
        name="di_b_jet_delta_phi",
        expression="di_b_jet.deltaphi",
        null_value=EMPTY_FLOAT,
        binning=(12, -6, 6),
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
        binning=(15, 0, 5),
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
        binning=(15, 0, 1),
        x_title=r"raw DeepJetFlawB",
    )

    cfg.add_variable(
        name="jet_raw_PNetB",
        expression="Jet.btagPNetB",
        null_value=EMPTY_FLOAT,
        binning=(15, 0, 1),
        x_title=r"raw PNetB",
    )


# =============================================================================
# High-level features
# =============================================================================

def add_highlevel_features(cfg: od.Config) -> None:
    """
    Adds MET and other high-level features.
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
        binning=(30, 0, 300),
        unit="GeV",
        x_title=r"PUPPI MET $p_T$",
    )

    cfg.add_variable(
        name="puppi_met_pt_recoil_corr",
        expression="RecoilCorrMET.pt",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 300),
        unit="GeV",
        x_title=r"RecoilCorrMET $p_T$",
    )

    cfg.add_variable(
        name="puppi_met_phi",
        expression="PuppiMET.phi",
        null_value=EMPTY_FLOAT,
        binning=(16, -3.2, 3.2),
        x_title=r"PUPPI MET $\phi$",
    )

    cfg.add_variable(
        name="D_zeta",
        expression="D_zeta",
        null_value=EMPTY_FLOAT,
        binning=(12, -80, 300),
        x_title="$D_{\\zeta}$",
    )

    cfg.add_variable(
        name="pt_H",
        expression="pt_H",
        null_value=EMPTY_FLOAT,
        binning=(12, 0, 250),
        x_title="$p_{T}(H)$",
    )


# =============================================================================
# Weight features
# =============================================================================

def add_weight_features(cfg: od.Config) -> None:
    """
    Adds weights.
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
        binning=(30, 0, 3),
        unit="",
        x_title=r"Pileup weight",
    )

    cfg.add_variable(
        name="muon_weight",
        expression="muon_weight_nom",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.5, 1.5),
        unit="",
        x_title=r"muon weight",
    )

    cfg.add_variable(
        name="tau_weight",
        expression="tau_weight_nom",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.5, 1.5),
        unit="",
        x_title=r"tau weight",
    )

    for var in [
        "0",
        "0p25",
        "0p375",
        "0p5",
        "minus0p25",
    ]:
        angle = (
            float(
                var
                .replace("minus", "-")
                .replace("p", ".")
            )
            * 180
        )

        cfg.add_variable(
            name=f"TauSpinner_weight_cp_{var}",
            expression=f"TauSpinner.weight_cp_{var}",
            null_value=EMPTY_FLOAT,
            binning=(60, -3, 3),
            unit="",
            x_title=(
                fr"Tau spinner weight "
                fr"$\Delta \phi$=${angle}^{{\circ}}$"
            ),
        )


# =============================================================================
# Cutflow
# =============================================================================

def add_cutflow_features(cfg: od.Config) -> None:
    """
    Adds cutflow features.
    """
    cfg.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )


# =============================================================================
# phi_CP
# =============================================================================

def phi_cp_variables(cfg: od.Config) -> None:
    n_bins_phi_cp = 11

    for the_ch in [
        "mu_pi",
        "mu_rho",
        "mu_a1_1pr",
        "rho_rho",
        "pi_pi",
    ]:
        spitted_str = the_ch.split("_")

        if "a1" in the_ch:
            title_str = (
                "\\"
                + spitted_str[0]
                + fr" a_1, {spitted_str[2]}"
            )
        else:
            title_str = (
                "\\"
                + spitted_str[0]
                + "\\"
                + spitted_str[1]
            )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}",
            expression=f"phi_cp_{the_ch}",
            null_value=EMPTY_FLOAT,
            binning=(
                n_bins_phi_cp,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}]$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg1",
            expression=f"phi_cp_{the_ch}_reg1",
            null_value=EMPTY_FLOAT,
            binning=(
                n_bins_phi_cp,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}], "
                rf"\alpha < \pi/4$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg2",
            expression=f"phi_cp_{the_ch}_reg2",
            null_value=EMPTY_FLOAT,
            binning=(
                n_bins_phi_cp,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}], "
                rf"\alpha \geq \pi/4$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}_2bin",
            expression=f"phi_cp_{the_ch}_2bin",
            null_value=EMPTY_FLOAT,
            binning=(
                2,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}]$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg1_2bin",
            expression=f"phi_cp_{the_ch}_reg1_2bin",
            null_value=EMPTY_FLOAT,
            binning=(
                2,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}], "
                rf"\alpha < \pi/4$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"phi_cp_{the_ch}_reg2_2bin",
            expression=f"phi_cp_{the_ch}_reg2_2bin",
            null_value=EMPTY_FLOAT,
            binning=(
                2,
                0,
                2 * np.pi,
            ),
            x_title=(
                rf"$\varphi_{{CP}} "
                rf"[{title_str}], "
                rf"\alpha \geq \pi/4$ (rad)"
            ),
        )

        cfg.add_variable(
            name=f"alpha_{the_ch}",
            expression=f"alpha_{the_ch}",
            null_value=EMPTY_FLOAT,
            binning=(
                6,
                0,
                np.pi / 2,
            ),
            x_title=(
                rf"$ \alpha "
                rf"[{title_str}] $(rad)"
            ),
        )


# =============================================================================
# Dilepton features
# =============================================================================

def add_dilepton_features(cfg: od.Config) -> None:
    channels = cfg.channels.names()

    ch_objects = DotDict.wrap({
        "etau": {
            "lep0": "Electron",
            "lep1": "Tau",
        },
        "mutau": {
            "lep0": "Muon",
            "lep1": "Tau",
        },
        "emu": {
            "lep0": "Electron",
            "lep1": "Muon",
        },
        "tautau": {
            "lep0": "Tau",
            "lep1": "Tau",
        },
    })

    # Used to define histograms for kinematic variables with finer binning
    bin_split_factor = 4

    for ch_str in channels:
        cfg.add_variable(
            name=f"{ch_str}_mvis",
            expression=f"hcand_{ch_str}.mass",
            null_value=EMPTY_FLOAT,
            binning=(25, 0.0, 250.0),
            unit="GeV",
            x_title=r"$m_{vis}$",
        )

        if ch_str in [
            "etau",
            "mutau",
        ]:
            cfg.add_variable(
                name=f"{ch_str}_mt",
                expression=f"hcand_{ch_str}.mt",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 200.0),
                unit="GeV",
                x_title="$\\m_{T}$",
            )

        if ch_str == "emu":
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

        for lep in [
            "lep0",
            "lep1",
        ]:
            if ch_str != "tautau":
                lep_str = ch_objects[ch_str][lep].lower()
            else:
                lep_str = f"tau {lep[3:]}"

            cfg.add_variable(
                name=f"{ch_str}_{lep}_pt",
                expression=f"hcand_{ch_str}.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(20, 15, 215),
                unit="GeV",
                x_title=rf"{lep_str} $p_{{T}}$",
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
                binning=(12, 0, 12),
                unit="",
                x_title=rf"{lep_str} PNet decay mode",
            )

            cfg.add_variable(
                name=f"{ch_str}_{lep}_decayMode",
                expression=f"hcand_{ch_str}.{lep}.decayMode",
                null_value=EMPTY_FLOAT,
                binning=(12, 0, 12),
                unit="",
                x_title=rf"{lep_str} HPS decay mode",
            )

            cfg.add_variable(
                name=f"{ch_str}_{lep}_ip_sig",
                expression=f"hcand_{ch_str}.{lep}.ip_sig",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 10),
                unit="",
                x_title=(
                    rf"{lep_str} "
                    rf"$\frac{{|IP|}}{{\sigma(IP)}}$"
                ),
            )

            for proj in [
                "x",
                "y",
                "z",
            ]:
                cfg.add_variable(
                    name=f"{ch_str}{lep}_ip_{proj}",
                    expression=f"hcand_{ch_str}.{lep}.IP{proj}",
                    null_value=EMPTY_FLOAT,
                    binning=(30, -0.002, 0.002),
                    unit="",
                    x_title=rf"{lep_str} $IP_{proj}$",
                )

            # Variables with finer binning
            cfg.add_variable(
                name=f"{ch_str}_{lep}_pt_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(
                    30 * bin_split_factor,
                    20,
                    80.0,
                ),
                unit="GeV",
                x_title=rf"{lep_str} $p_{{T}}$",
            )

            cfg.add_variable(
                name=f"{ch_str}_{lep}_eta_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.eta",
                null_value=EMPTY_FLOAT,
                binning=(
                    32 * bin_split_factor,
                    -3.2,
                    3.2,
                ),
                x_title=rf"{lep_str} $\eta$",
            )

            cfg.add_variable(
                name=f"{ch_str}_{lep}_phi_fine_binning",
                expression=f"hcand_{ch_str}.{lep}.phi",
                null_value=EMPTY_FLOAT,
                binning=(
                    32 * bin_split_factor,
                    -3.2,
                    3.2,
                ),
                x_title=rf"{lep_str} $\phi$",
            )

            # FastMTT variables
            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_px",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.px",
                null_value=EMPTY_FLOAT,
                binning=(42, -10.0, 200.0),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{x}^{fastMTT}$",
            )

            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_py",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.py",
                null_value=EMPTY_FLOAT,
                binning=(42, -10.0, 200.0),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{y}^{fastMTT}$",
            )

            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_pz",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.pz",
                null_value=EMPTY_FLOAT,
                binning=(42, -10.0, 200.0),
                unit="GeV",
                x_title=f"{lep} " + r"$p_{z}^{fastMTT}$",
            )

            cfg.add_variable(
                name=f"hcand_{ch_str}_fastMTT_{lep}_pt",
                expression=f"hcand_{ch_str}.fastMTT.{lep}.pt",
                null_value=EMPTY_FLOAT,
                binning=(40, 0.0, 200.0),
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

BDT_CLASS_LABELS = (
    "ggphi",
    "bbphi",
    "dy",
    "tt",
)

BDT_CLASS_TITLES = {
    "ggphi": r"gg$\phi$($\phi\rightarrow\tau\tau$)",
    "bbphi": r"bb$\phi$($\phi\rightarrow\tau\tau$)",
    "dy": "DY",
    "tt": r"t$\bar{t}$",
}


# Discriminants with their own adaptive binning.
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


# Diagnostic signal-splitting discriminants.
BDT_DIAGNOSTIC_1D_DISCRIMINANTS = {
    "D_ggphi_sig": {
        "title": (
            r"$P_{\mathrm{gg}\phi}/"
            r"(P_{\mathrm{gg}\phi}+P_{\mathrm{bb}\phi})$"
        ),
        "binning_source": "Disc_ggphi",
    },

    "D_bbphi_sig": {
        "title": (
            r"$P_{\mathrm{bb}\phi}/"
            r"(P_{\mathrm{gg}\phi}+P_{\mathrm{bb}\phi})$"
        ),
        "binning_source": "Disc_bbphi",
    },
}


# Backward-compatible aliases.
BDT_DISCRIMINANT_ALIASES = {
    "D_dy": "D_DY",
    "D_tt": "D_TT",
}


# Flattened 2D variables.
BDT_2D_FLATTENED_PAIRS = (
    (
        "D_sig_vs_Disc_ggphi",
        "D_sig",
        "Disc_ggphi",
    ),

    (
        "D_sig_vs_Disc_bbphi",
        "D_sig",
        "Disc_bbphi",
    ),

    (
        "D_sig_vs_D_ggphi",
        "D_sig",
        "D_ggphi",
    ),

    (
        "D_sig_vs_D_bbphi",
        "D_sig",
        "D_bbphi",
    ),

    (
        "D_ggphi_vs_D_bbphi",
        "D_ggphi",
        "D_bbphi",
    ),
)


# =============================================================================
# Optimized irregular 2D variables
# =============================================================================
#
# These two pairs MUST use the dedicated y-merged optimization JSON.
#
# There is deliberately no rectangular fallback for these.
#
# Example for M100 D_sig vs Disc_ggphi:
#
#   n_y_bins_by_x = [5, 6, 7, 8, 8, 8, 8, 6]
#
#   n_cells =
#       5 + 6 + 7 + 8 + 8 + 8 + 8 + 6
#       = 56
#
# Therefore the flattened histogram has exactly 56 bins.
#
BDT_2D_YMERGED_PAIRS = {
    (
        "D_sig",
        "Disc_ggphi",
    ),
    (
        "D_sig",
        "Disc_bbphi",
    ),
}


def _bdt_all_1d_discriminants() -> dict:
    out = {}

    out.update(
        BDT_1D_DISCRIMINANTS
    )

    out.update(
        BDT_DIAGNOSTIC_1D_DISCRIMINANTS
    )

    return out


def _bdt_discriminant_binning_source(
    discriminant: str,
) -> str:
    """
    Return the discriminant whose adaptive binning should be used.
    """
    all_discriminants = (
        _bdt_all_1d_discriminants()
    )

    if discriminant in all_discriminants:
        return (
            all_discriminants[
                discriminant
            ][
                "binning_source"
            ]
        )

    if discriminant in BDT_DISCRIMINANT_ALIASES:
        return (
            BDT_DISCRIMINANT_ALIASES[
                discriminant
            ]
        )

    return discriminant


def _bdt_n_bins_from_binning(
    binning,
) -> int:
    """
    Return number of bins from either:

      - regular binning:
            (n_bins, x_min, x_max)

      - variable binning:
            [edge0, edge1, ...]
    """
    b = list(
        binning
    )

    if (
        len(b) == 3
        and isinstance(
            b[0],
            (
                int,
                np.integer,
            ),
        )
        and b[0] > 0
        and float(b[1]) < float(b[2])
    ):
        return int(
            b[0]
        )

    return (
        len(b)
        - 1
    )


def _bdt_binning(
    mass: int,
    discriminant: str,
):
    """
    Return adaptive binning for one BDT output variable.
    """
    source = (
        _bdt_discriminant_binning_source(
            discriminant
        )
    )

    return (
        _read_bdt_adaptive_binning(
            mass,
            source,
        )
    )


# =============================================================================
# Optimized y-merged 2D binning
# =============================================================================

def _bdt_ymerged_2d_edges_path(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
) -> Path:
    """
    Return the optimized y-merged 2D JSON path.

    Example:

      x = D_sig
      y = Disc_ggphi
      M = 100

    gives

      2D_Disc_ggphi_vs_D_sig_
      yMergedMinWeightedBkg_edges_
      M100_combined_crossApplied.json
    """
    mass = int(
        mass
    )

    safe_pair = (
        f"{y_discriminant}_vs_{x_discriminant}"
        .replace(
            "/",
            "_",
        )
    )

    return (
        BDT_OUTPUT_BASE
        / f"M{mass}"
        / "adaptive_discriminant_rebinning"
        / f"M{mass}"
        / "two_dimensional_discriminants"
        / BDT_2D_ADAPTIVE_SUBDIR
        / (
            f"2D_{safe_pair}_"
            f"{BDT_2D_YMERGED_TOKEN}_"
            f"edges_M{mass}_"
            f"{BDT_ADAPTIVE_TAG}.json"
        )
    )


def _bdt_extract_ymerged_2d_binning_from_json(
    data,
) -> dict:
    """
    Extract and strictly validate the optimized irregular 2D binning.

    Expected JSON structure:

        {
            "x_edges": [...],

            "y_edges_by_x_bin": [
                {
                    "x_bin": 1,
                    "x_low": ...,
                    "x_high": ...,
                    "y_edges": [...]
                },
                ...
            ],

            "n_cells": ...
        }

    Flattening convention:

        flat_index =
            x_bin_offset[ix] + iy

    where:

        x_bin_offset[ix] =
            sum(
                n_y_bins
                in all previous x bins
            )

    Hence there is exactly one flattened bin per optimized irregular 2D
    cell.
    """
    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "Optimized 2D binning JSON payload "
            "is not a dictionary"
        )

    if "x_edges" not in data:
        raise KeyError(
            "Could not find 'x_edges' "
            "in optimized 2D JSON payload"
        )

    if "y_edges_by_x_bin" not in data:
        raise KeyError(
            "Could not find 'y_edges_by_x_bin' "
            "in optimized 2D JSON payload"
        )

    x_edges = np.asarray(
        [
            float(x)
            for x in data[
                "x_edges"
            ]
        ],
        dtype=np.float64,
    )

    if (
        x_edges.ndim != 1
        or len(x_edges) < 2
    ):
        raise ValueError(
            "Invalid optimized 2D "
            f"x_edges: {x_edges}"
        )

    if not np.all(
        np.diff(
            x_edges
        ) > 0.0
    ):
        raise ValueError(
            "Optimized 2D x edges are "
            "not strictly increasing: "
            f"{x_edges}"
        )

    y_entries = (
        data[
            "y_edges_by_x_bin"
        ]
    )

    if not isinstance(
        y_entries,
        list,
    ):
        raise TypeError(
            "Invalid optimized 2D binning: "
            "y_edges_by_x_bin is not a list"
        )

    n_x_bins = (
        len(x_edges)
        - 1
    )

    if len(y_entries) != n_x_bins:
        raise ValueError(
            "Invalid optimized 2D binning: "
            f"len(y_edges_by_x_bin)={len(y_entries)} "
            f"but n_x_bins={n_x_bins}"
        )

    y_edges_by_x_bin = []
    n_y_bins_by_x_bin = []
    x_bin_offsets = []

    offset = 0

    for ix, entry in enumerate(
        y_entries
    ):
        if not isinstance(
            entry,
            dict,
        ):
            raise TypeError(
                "Invalid optimized 2D binning: "
                f"entry for x bin {ix} "
                "is not a dictionary"
            )

        if "y_edges" not in entry:
            raise KeyError(
                "Invalid optimized 2D binning: "
                f"missing y_edges for x bin {ix}"
            )

        y_edges = np.asarray(
            [
                float(y)
                for y in entry[
                    "y_edges"
                ]
            ],
            dtype=np.float64,
        )

        if (
            y_edges.ndim != 1
            or len(y_edges) < 2
        ):
            raise ValueError(
                "Invalid optimized y edges "
                f"for x bin {ix}: "
                f"{y_edges}"
            )

        if not np.all(
            np.diff(
                y_edges
            ) > 0.0
        ):
            raise ValueError(
                "Optimized y edges are not "
                "strictly increasing "
                f"for x bin {ix}: "
                f"{y_edges}"
            )

        n_y_bins = (
            len(y_edges)
            - 1
        )

        x_bin_offsets.append(
            int(offset)
        )

        y_edges_by_x_bin.append(
            y_edges.tolist()
        )

        n_y_bins_by_x_bin.append(
            int(n_y_bins)
        )

        offset += (
            n_y_bins
        )

    n_cells_from_edges = int(
        offset
    )

    n_cells_declared = (
        data.get(
            "n_cells"
        )
    )

    if n_cells_declared is not None:
        n_cells_declared = int(
            n_cells_declared
        )

        if (
            n_cells_declared
            != n_cells_from_edges
        ):
            raise ValueError(
                "Invalid optimized 2D binning: "
                f"JSON declares n_cells={n_cells_declared}, "
                f"but y_edges_by_x_bin gives "
                f"{n_cells_from_edges}"
            )

    if n_cells_from_edges <= 0:
        raise ValueError(
            "Invalid optimized 2D binning: "
            "zero flattened cells"
        )

    return {
        "x_edges":
            x_edges.tolist(),

        "y_edges_by_x_bin":
            y_edges_by_x_bin,

        "n_y_bins_by_x_bin":
            n_y_bins_by_x_bin,

        "x_bin_offsets":
            x_bin_offsets,

        "n_cells":
            n_cells_from_edges,
    }


def _bdt_extract_n_flat_bins_from_ymerged_2d_json(
    data,
) -> int:
    """
    Extract the total number of optimized flattened cells.
    """
    optimized = (
        _bdt_extract_ymerged_2d_binning_from_json(
            data
        )
    )

    return int(
        optimized[
            "n_cells"
        ]
    )


def _read_bdt_ymerged_2d_flattened_binning(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
):
    """
    Return flattened histogram binning for an optimized y-merged pair.

    For pairs listed in BDT_2D_YMERGED_PAIRS, the optimization JSON is
    MANDATORY.

    There is deliberately NO rectangular fallback.

    For example, if the optimized M100 D_sig vs Disc_ggphi JSON contains:

        n_cells = 56

    this function returns:

        (56, 0, 56)

    It can never silently become:

        (900, 0, 900)
    """
    pair = (
        str(
            x_discriminant
        ),
        str(
            y_discriminant
        ),
    )

    if pair not in BDT_2D_YMERGED_PAIRS:
        return None

    path = (
        _bdt_ymerged_2d_edges_path(
            mass,
            x_discriminant,
            y_discriminant,
        )
    )

    if not path.is_file():
        raise RuntimeError(
            "\n"
            "Missing mandatory optimized y-merged "
            "2D BDT binning JSON.\n"
            "\n"
            f"Mass           : {mass}\n"
            f"x discriminant : {x_discriminant}\n"
            f"y discriminant : {y_discriminant}\n"
            f"BDT base       : {BDT_OUTPUT_BASE}\n"
            f"Expected JSON  : {path}\n"
            "\n"
            "Rectangular fallback is intentionally disabled.\n"
            "\n"
            "Make the optimized training output available and/or set:\n"
            "\n"
            "  export MSSM_BDT_BINNING_BASE=/path/to/"
            "bdt_3_classes_10_features_clippedJetCounts\n"
        )

    try:
        data = json.loads(
            path.read_text()
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to read mandatory optimized "
            "y-merged 2D BDT binning JSON:\n"
            f"  {path}"
        ) from exc

    # ---------------------------------------------------------------------
    # Optional training-metadata validation.
    # ---------------------------------------------------------------------

    json_x_name = (
        data.get(
            "x_name"
        )
    )

    json_y_name = (
        data.get(
            "y_name"
        )
    )

    if (
        json_x_name is not None
        and str(json_x_name)
        != str(x_discriminant)
    ):
        raise RuntimeError(
            "Wrong x discriminant in optimized "
            f"2D JSON:\n  {path}\n"
            f"Found    : {json_x_name}\n"
            f"Expected : {x_discriminant}"
        )

    if (
        json_y_name is not None
        and str(json_y_name)
        != str(y_discriminant)
    ):
        raise RuntimeError(
            "Wrong y discriminant in optimized "
            f"2D JSON:\n  {path}\n"
            f"Found    : {json_y_name}\n"
            f"Expected : {y_discriminant}"
        )

    try:
        optimized = (
            _bdt_extract_ymerged_2d_binning_from_json(
                data
            )
        )

    except Exception as exc:
        raise RuntimeError(
            "Invalid mandatory optimized "
            "y-merged 2D BDT binning JSON:\n"
            f"  {path}"
        ) from exc

    n_flat_bins = int(
        optimized[
            "n_cells"
        ]
    )

    return (
        n_flat_bins,
        0,
        n_flat_bins,
    )


def _bdt_flattened_2d_binning(
    mass: int,
    x_discriminant: str,
    y_discriminant: str,
):
    """
    Return flattened 2D histogram binning.

    Optimized pairs:

        D_sig vs Disc_ggphi
        D_sig vs Disc_bbphi

    MUST use their dedicated irregular/y-merged optimization JSON.

    Other diagnostic 2D variables continue to use rectangular binning based
    on their ordinary adaptive 1D edges.
    """
    pair = (
        str(
            x_discriminant
        ),
        str(
            y_discriminant
        ),
    )

    # ---------------------------------------------------------------------
    # Optimized datacard 2D discriminants.
    #
    # This is deliberately strict.
    # There is no fallback to n_x * n_y.
    # ---------------------------------------------------------------------

    if pair in BDT_2D_YMERGED_PAIRS:
        return (
            _read_bdt_ymerged_2d_flattened_binning(
                mass,
                x_discriminant,
                y_discriminant,
            )
        )

    # ---------------------------------------------------------------------
    # Other diagnostic 2D variables remain rectangular.
    # ---------------------------------------------------------------------

    x_binning = (
        _bdt_binning(
            mass,
            x_discriminant,
        )
    )

    y_binning = (
        _bdt_binning(
            mass,
            y_discriminant,
        )
    )

    n_x_bins = (
        _bdt_n_bins_from_binning(
            x_binning
        )
    )

    n_y_bins = (
        _bdt_n_bins_from_binning(
            y_binning
        )
    )

    n_flat_bins = (
        n_x_bins
        * n_y_bins
    )

    return (
        n_flat_bins,
        0,
        n_flat_bins,
    )


def _bdt_discriminant_title(
    discriminant: str,
) -> str:
    """
    Return readable title for a BDT discriminant.
    """
    all_discriminants = (
        _bdt_all_1d_discriminants()
    )

    if discriminant in all_discriminants:
        return (
            all_discriminants[
                discriminant
            ][
                "title"
            ]
        )

    if discriminant in BDT_DISCRIMINANT_ALIASES:
        target = (
            BDT_DISCRIMINANT_ALIASES[
                discriminant
            ]
        )

        return (
            all_discriminants[
                target
            ][
                "title"
            ]
        )

    return str(
        discriminant
    )


# =============================================================================
# Register MSSM BDT variables
# =============================================================================

def add_mssm_bdt_output(
    cfg: od.Config,
) -> None:
    """
    Register all per-mass BDT outputs.

    The two main signal-like flattened discriminants:

        bdt_D_sig_vs_Disc_ggphi_M{mass}
        bdt_D_sig_vs_Disc_bbphi_M{mass}

    always use the dedicated optimized y-merged 2D JSON.

    If the optimized JSON is unavailable, config construction intentionally
    fails instead of silently switching to rectangular binning.
    """
    from MSSM_H_tt.config.mass_points import (
        read_bdt_masses,
    )

    MASS_POINTS = (
        read_bdt_masses()
    )

    for m in MASS_POINTS:

        # -----------------------------------------------------------------
        # Raw four-class probabilities
        # -----------------------------------------------------------------

        for label in BDT_CLASS_LABELS:
            cfg.add_variable(
                name=(
                    f"bdt_raw_score_"
                    f"{label}_M{m}"
                ),
                expression=(
                    f"bdt_raw_score_"
                    f"{label}_M{m}"
                ),
                null_value=EMPTY_FLOAT,
                binning=BDT_DEFAULT_SCORE_BINNING,
                x_title=(
                    f"BDT probability for "
                    f"{BDT_CLASS_TITLES[label]} "
                    f"(M={m} GeV)"
                ),
            )

        # -----------------------------------------------------------------
        # Standard and diagnostic 1D BDT discriminants
        # -----------------------------------------------------------------

        for (
            discr_name,
            discr_info,
        ) in _bdt_all_1d_discriminants().items():

            cfg.add_variable(
                name=(
                    f"bdt_"
                    f"{discr_name}_"
                    f"M{m}"
                ),
                expression=(
                    f"bdt_"
                    f"{discr_name}_"
                    f"M{m}"
                ),
                null_value=EMPTY_FLOAT,
                binning=(
                    _bdt_binning(
                        m,
                        discr_name,
                    )
                ),
                x_title=(
                    f"{discr_info['title']} "
                    f"(M={m} GeV)"
                ),
            )

        # -----------------------------------------------------------------
        # Backward-compatible aliases
        # -----------------------------------------------------------------

        for (
            alias,
            target,
        ) in BDT_DISCRIMINANT_ALIASES.items():

            cfg.add_variable(
                name=(
                    f"bdt_"
                    f"{alias}_"
                    f"M{m}"
                ),
                expression=(
                    f"bdt_"
                    f"{target}_"
                    f"M{m}"
                ),
                null_value=EMPTY_FLOAT,
                binning=(
                    _bdt_binning(
                        m,
                        target,
                    )
                ),
                x_title=(
                    f"{_bdt_discriminant_title(target)} "
                    f"(M={m} GeV)"
                ),
            )

        # -----------------------------------------------------------------
        # Flattened 2D BDT variables
        # -----------------------------------------------------------------

        for (
            pair_name,
            x_discr,
            y_discr,
        ) in BDT_2D_FLATTENED_PAIRS:

            cfg.add_variable(
                name=(
                    f"bdt_"
                    f"{pair_name}_"
                    f"M{m}"
                ),
                expression=(
                    f"bdt_"
                    f"{pair_name}_"
                    f"M{m}"
                ),
                null_value=EMPTY_FLOAT,
                binning=(
                    _bdt_flattened_2d_binning(
                        m,
                        x_discr,
                        y_discr,
                    )
                ),
                x_title=(
                    "Flattened 2D bin: "
                    f"{_bdt_discriminant_title(x_discr)} "
                    "vs "
                    f"{_bdt_discriminant_title(y_discr)} "
                    f"(M={m} GeV)"
                ),
            )

        # -----------------------------------------------------------------
        # Four BDT regions
        # -----------------------------------------------------------------

        cfg.add_variable(
            name=f"bdt_cat_M{m}",
            expression=f"bdt_cat_M{m}",
            binning=(
                4,
                -0.5,
                3.5,
            ),
            discrete_x=True,
            x_title=(
                r"BDT category: "
                r"0=gg$\phi$, "
                r"1=bb$\phi$, "
                r"2=DY, "
                r"3=t$\bar{t}$ "
                f"(M={m} GeV)"
            ),
        )


# =============================================================================
# e-mu phi_CP features
# =============================================================================

def add_emu_phi_cp_features(
    cfg: od.Config,
) -> None:

    cfg.add_variable(
        name="phi_cp_emu",
        expression="phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(
            16,
            0.0,
            2 * np.pi,
        ),
        x_title=r"$\varphi_{CP}^{e\mu}$ (rad)",
    )

    cfg.add_variable(
        name="cos_phi_cp_emu",
        expression="cos_phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(
            20,
            -1.0,
            1.0,
        ),
        x_title=r"$\cos(\varphi_{CP}^{e\mu})$",
    )

    cfg.add_variable(
        name="sin_phi_cp_emu",
        expression="sin_phi_cp_emu",
        null_value=EMPTY_FLOAT,
        binning=(
            20,
            -1.0,
            1.0,
        ),
        x_title=r"$\sin(\varphi_{CP}^{e\mu})$",
    )


# =============================================================================
# Main variable registration
# =============================================================================

def add_variables(
    cfg: od.Config,
) -> None:
    """
    Adds all variables to a config.
    """
    add_common_features(
        cfg
    )

    add_lepton_features(
        cfg
    )

    add_jet_features(
        cfg
    )

    add_highlevel_features(
        cfg
    )

    add_weight_features(
        cfg
    )

    add_cutflow_features(
        cfg
    )

    add_dilepton_features(
        cfg
    )

    add_mssm_bdt_output(
        cfg
    )

    add_emu_phi_cp_features(
        cfg
    )