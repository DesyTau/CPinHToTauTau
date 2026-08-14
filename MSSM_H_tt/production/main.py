"""
Column production methods related to higher-level features.
"""
import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.reduction.util import create_collections_from_masks
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.reduction.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.columnar_util import optional_column as optional
from columnflow.production.util import attach_coffea_behavior

from MSSM_H_tt.production.pileup import pu_weight
from MSSM_H_tt.production.weights import muon_weight, tau_weight, electron_weight, trigger_sf
from MSSM_H_tt.production.sample_split import split_dy
from MSSM_H_tt.production.generatorZ import generatorZ
from MSSM_H_tt.production.dilepton_features import hcand_fields, hcand_mt
from MSSM_H_tt.production.z_pt_reweighting import zpt_weight
from MSSM_H_tt.production.aux_columns import (
    jet_pt_def,
    jets_taggable,
    number_b_jet,
    create_jetID_masks,
)
from MSSM_H_tt.production.btag_SF import btag_weight_SF
from MSSM_H_tt.production.top_pt_weight import top_pt_weight, gen_parton_top
from MSSM_H_tt.production.recoil_corr import gen_dilepton, recoil_corrected_met
from MSSM_H_tt.production.bdt_score import mssm_bdt_score
from MSSM_H_tt.production.fastMTT import fastMTT
from MSSM_H_tt.production.pt_H import pt_H
from MSSM_H_tt.production.D_zeta import D_zeta
from MSSM_H_tt.production.stitching_weights import stitching_weight
from MSSM_H_tt.production.unclustered_met import unclustered_met, add_unclustered_to_recoilcorrmet
from MSSM_H_tt.production.theor_weight import theor_unc
from MSSM_H_tt.production.bdt_2d_bins import bdt_2d_variables
np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

from columnflow.columnar_util import has_ak_column

def _copy_recoilcorrmet(events, suffix, pt, phi, covXX, covXY, covYY):
    events = set_ak_column(
        events,
        f"RecoilCorrMET.pt{suffix}",
        pt,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        f"RecoilCorrMET.phi{suffix}",
        phi,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        f"RecoilCorrMET.covXX",
        covXX,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        f"RecoilCorrMET.covXY",
        covXY,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        f"RecoilCorrMET.covYY",
        covYY,
        value_type=np.float32,
    )
    return events


def build_recoilcorrmet_passthrough(events: ak.Array) -> ak.Array:
    events = _copy_recoilcorrmet(
        events,
        "",
        events.PuppiMET.pt,
        events.PuppiMET.phi,
        events.PuppiMET.covXX,
        events.PuppiMET.covXY,
        events.PuppiMET.covYY
    )

    for direction in ("up", "down"):
        pt_col = f"PuppiMET.pt_unclustered_{direction}"
        phi_col = f"PuppiMET.phi_unclustered_{direction}"

        if has_ak_column(events, pt_col):
            pt = events.PuppiMET[f"pt_unclustered_{direction}"]
            phi = (
                events.PuppiMET[f"phi_unclustered_{direction}"]
                if has_ak_column(events, phi_col)
                else events.PuppiMET.phi
            )

            events = _copy_recoilcorrmet(
                events,
                f"_unclustered_{direction}",
                pt,
                phi,
                events.PuppiMET.covXX,
                events.PuppiMET.covXY,
                events.PuppiMET.covYY,
            )

    return events

@producer(
    uses={
        "event",
        attach_coffea_behavior,
        mc_weight,
        normalization_weights,
        split_dy,
        pu_weight,
        muon_weight,
        tau_weight,
        electron_weight,
        trigger_sf,
        generatorZ,
        zpt_weight,
        hcand_fields,
        hcand_mt,
        number_b_jet,
        create_jetID_masks,
        jet_pt_def,
        jets_taggable,
        btag_weight_SF,
        gen_parton_top,
        top_pt_weight,
        gen_dilepton,
        recoil_corrected_met,
        trigger_sf,
        fastMTT,
        pt_H,
        D_zeta,
        stitching_weight,
        unclustered_met,
        theor_unc,
        "PuppiMET.pt", "PuppiMET.phi", "PuppiMET.covXX", "PuppiMET.covXY", "PuppiMET.covYY",
        add_unclustered_to_recoilcorrmet,
    },
    produces={
        "event",
        attach_coffea_behavior,
        mc_weight,
        normalization_weights,
        split_dy,
        pu_weight,
        muon_weight,
        tau_weight,
        electron_weight,
        trigger_sf,
        generatorZ,
        zpt_weight,
        hcand_fields,
        hcand_mt,
        # jet producers
        number_b_jet,
        create_jetID_masks,
        jet_pt_def,
        jets_taggable,

        # explicit BDT input columns produced by jet_pt_def and number_b_jet
        "mt_jets",
        "mt_bjets",
        "n_jets_clipped",
        "n_bjets_clipped",

        btag_weight_SF,
        gen_parton_top,
        top_pt_weight,
        gen_dilepton,
        recoil_corrected_met,
        trigger_sf,
        fastMTT,
        pt_H,
        D_zeta,
        stitching_weight,
        unclustered_met,
        theor_unc,
        "RecoilCorrMET.pt", "RecoilCorrMET.phi",
        "RecoilCorrMET.covXX", "RecoilCorrMET.covXY", "RecoilCorrMET.covYY",
        "RecoilCorrMET.pt_unclustered_up",
        "RecoilCorrMET.phi_unclustered_up",
        "RecoilCorrMET.pt_unclustered_down",
        "RecoilCorrMET.phi_unclustered_down",
        add_unclustered_to_recoilcorrmet,
        },
    produce_weights=True,
)
def main_common(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    processes = self.dataset_inst.processes.names()

    events = self[attach_coffea_behavior](events, **kwargs)

    # ------------------------------------------------------------
    # Basic object and candidate features
    # ------------------------------------------------------------
    print("Producing jet variables for plotting...")
    events = self[create_jetID_masks](events, **kwargs)
    events = self[jet_pt_def](events, **kwargs)
    events = self[jets_taggable](events, **kwargs)
    
    events = self[unclustered_met](events, **kwargs)

    met_recoil_datasets = self.config_inst.x.met_recoil.datasets
    dataset_name = self.dataset_inst.name

    if self.dataset_inst.is_mc and dataset_name in met_recoil_datasets:
        print(f"Producing gen dilepton information for MET recoil for {dataset_name}...")
        events = self[gen_dilepton](events, **kwargs)

        print(
            f"Producing recoil-corrected MET and MET recoil variations "
            f"for {dataset_name} with order={met_recoil_datasets[dataset_name]}..."
        )
        events = self[recoil_corrected_met](events, **kwargs)
        # Add unclustered MET variations on top of recoil-corrected MET
        print("Add unclustered MET variations on top of recoil-corrected MET")
        events = self[add_unclustered_to_recoilcorrmet](events, **kwargs)

    else:
        if self.dataset_inst.is_mc:
            print(
                f"Dataset {dataset_name} not in cfg.x.met_recoil.datasets. "
                "Creating RecoilCorrMET passthrough from PuppiMET..."
            )
        else:
            print("Creating RecoilCorrMET passthrough for data...")

        events = build_recoilcorrmet_passthrough(events)

    print("Producing Hcand features...")
    events = self[hcand_fields](events, **kwargs)

    print("Producing Number of b-jets for categorization...")
    events = self[number_b_jet](events, **kwargs)

    # ------------------------------------------------------------
    # MET-dependent high-level variables
    #
    # These producers should read RecoilCorrMET, not PuppiMET.
    # ------------------------------------------------------------

    print(
    "PuppiMET unclustered up mean delta:",
    ak.mean(events.PuppiMET.pt_unclustered_up - events.PuppiMET.pt),
    )

    print(
        "RecoilCorrMET unclustered up mean delta:",
        ak.mean(events.RecoilCorrMET.pt_unclustered_up - events.RecoilCorrMET.pt),
    )

    print(
        "RecoilCorrMET unclustered down mean delta:",
        ak.mean(events.RecoilCorrMET.pt_unclustered_down - events.RecoilCorrMET.pt),
    )

    print("Producing mT distributions...")
    events = self[hcand_mt](events, **kwargs)
  
    print("Producing fastMTT features...")
    events = self[fastMTT](events, **kwargs)

    print("Producing pt_H features...")
    events = self[pt_H](events, **kwargs)

    print("Producing D_zeta features...")
    events = self[D_zeta](events, **kwargs)

    # ------------------------------------------------------------
    # Optional DY split
    # ------------------------------------------------------------
    if (self.dataset_inst.is_mc & (self.config_inst.channels.names()[0] != "emu")):
        if ak.any(["dy" in proc for proc in processes]):
            print("Splitting Drell-Yan dataset...")
            events = self[split_dy](events, **kwargs)

    # ------------------------------------------------------------
    # MC weights and systematics
    # ------------------------------------------------------------
    if self.dataset_inst.is_mc:

        print("Producing normalization weights...")
        events = self[mc_weight](events, **kwargs)
        events = self[normalization_weights](events, **kwargs)

        events = self[generatorZ](events, **kwargs)

        print("Z pt reweighting...")
        events = self[zpt_weight](events, **kwargs)

        print("Producing PU weights...")
        events = self[pu_weight](events, **kwargs)

        print("Producing Muon weights...")
        events = self[muon_weight](events, do_syst=True, **kwargs)

        print("Producing Electron weights...")
        events = self[electron_weight](events, do_syst=True, **kwargs)

        print("Producing SFs from efficiencies...")
        events = self[trigger_sf](events, **kwargs)

        print("Producing Tau weights...")
        events = self[tau_weight](events, do_syst=True, **kwargs)

        print("Producing btag SF fixed WP approach...")
        year = self.config_inst.x.year
        tag = self.config_inst.x.tag
        btag_wp = self.config_inst.x.btag_working_points[year][tag].particleNet.medium

        dis = events.Jet.btagPNetB
        nan_mask = np.isnan(dis)
        mask = ~np.isnan(dis)

        Jet = events.Jet[mask]

        jet_selections = {
            "jet_pt_20": Jet.pt > 20.0,
            "jet_eta_2.5": abs(Jet.eta) < 2.5,
            "jet_id": Jet.pass_tightID_lep_veto,
            "btag_wp_medium": Jet.btagPNetB >= btag_wp,
        }
        jet_obj_mask = ak.ones_like(Jet.pt, dtype=np.bool_)
        for the_sel in jet_selections.values():
            jet_obj_mask = jet_obj_mask & the_sel

        print("Producing btag SF weights...")
        events = self[btag_weight_SF](events, do_syst=True, **kwargs)

        print("Producing GenPartonTop...")
        events = self[gen_parton_top](events, **kwargs)

        top_pt_weight_dummy = ak.where(
            events.GenPartonTop.pt > 500.0,
            500.0,
            events.GenPartonTop.pt,
        )
        top_pt_weight_dummy = ak.ones_like(top_pt_weight_dummy)

        for variation in ("", "_up", "_down"):
            events = set_ak_column(
                events,
                f"top_pt_weight{variation}",
                top_pt_weight_dummy,
            )

        if (dataset_inst := getattr(self, "dataset_inst", None)) and dataset_inst.has_tag("ttbar"):
            print("Producing Top pT weights...")
            events = self[top_pt_weight](events, **kwargs)

        if self.dataset_inst.name in self.config_inst.x.stitch_samples:
            print("Producing stitching weights...")
            events = self[stitching_weight](events, **kwargs)
        else:
            events = set_ak_column_f32(
                events,
                "stitching_weight",
                ak.ones_like(events.event, dtype=np.float32),
            )
        
            
        events = self[theor_unc](events, **kwargs)

    return events

@main_common.init
def main_common_init(self: Producer) -> None:
    # main_common computes MET-dependent derived variables after recoil_corrected_met,
    # so main_common itself must run for MET and recoil shifted branches.
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("met") or shift_inst.has_tag("met_recoil")
    }

@producer(
    uses={
        main_common,
        mssm_bdt_score,
        bdt_2d_variables,
        category_ids,
    },
    produces={
        main_common,
        mssm_bdt_score,
        bdt_2d_variables,
        category_ids,
    },
    produce_weights=True,
)
def main(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Backward-compatible full producer used for plotting and
    workflows that require all BDT masses.
    """

    events = self[main_common](
        events,
        **kwargs,
    )
    print("Producing mssm_bdt_score...")

    events = self[mssm_bdt_score](
        events,
        **kwargs,
    )
    
    print("Producing bdt_2d_variables...")
    events = self[bdt_2d_variables](
        events,
        **kwargs,
    )
    
    print("Producing category_ids...")
    events = self[category_ids](
        events,
        **kwargs,
    )

    return events