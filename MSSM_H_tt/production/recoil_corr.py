# coding: utf-8

"""
Column production methods related to generator-level dileptons and MET recoil corrections.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def _status_flag(flags: ak.Array, bit: int) -> ak.Array:
    return (flags >> bit) & 1


def _sum_gen_p4(parts: ak.Array) -> ak.Array:
    """
    Properly sum a collection of generator-level particles as four-vectors.

    Do NOT sum pt/eta/phi/mass component-wise.  The boson pt/phi required by
    the recoil correction must come from the summed four-vector.
    """

    import vector

    ak.behavior.update(vector.backends.awkward.behavior)

    p4 = ak.zip(
        {
            "pt": parts.pt,
            "eta": parts.eta,
            "phi": parts.phi,
            "mass": parts.mass,
        },
        with_name="Momentum4D",
    )

    return ak.sum(p4, axis=-1)


def _safe_first(array: ak.Array, default: int = 0) -> ak.Array:
    return ak.fill_none(ak.firsts(array, axis=-1), default)


@producer(
    uses={"GenPart.*"},
    produces={
        "gen_dilepton_{pdgid,pt}",
        "gen_dilepton_{vis,all}.{pt,eta,phi,mass}",
    },
)
def gen_dilepton(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Reconstruct generator-level dilepton / boson information.

    Provides:
      - gen_dilepton_pdgid
      - gen_dilepton_pt
      - gen_dilepton_vis.{pt,eta,phi,mass}
      - gen_dilepton_all.{pt,eta,phi,mass}

    Important:
      - "all" includes neutrinos.
      - "vis" excludes neutrinos.
      - The boson kinematics are computed from a proper four-vector sum.
    """

    pdg_id = abs(events.GenPart.pdgId)
    status = events.GenPart.status
    flags = events.GenPart.statusFlags

    # Lepton masks for DY pTll-like quantities.
    ele_mu_mask = (
        ((pdg_id == 11) | (pdg_id == 13))
        & (status == 1)
        & (_status_flag(flags, 8) == 1)
    )

    tau_mask = (
        (pdg_id == 15)
        & (status == 2)
        & (_status_flag(flags, 8) == 1)
    )

    lepton_mask = ele_mu_mask | tau_mask

    # Recoil-correction boson decay products.
    #
    # Full boson:
    #   includes visible decay products and neutrinos.
    #
    # Visible boson:
    #   same collection, but neutrinos removed.
    #
    # The second OR term keeps direct prompt tau decay products, which is needed
    # for Z/H -> tautau final states.
    lepton_all_mask = (
        (
            (pdg_id >= 11)
            & (pdg_id <= 16)
            & (status == 1)
            & (_status_flag(flags, 8) == 1)
        )
        | (_status_flag(flags, 10) == 1)
    )

    neutrino_mask = (pdg_id == 12) | (pdg_id == 14) | (pdg_id == 16)
    lepton_vis_mask = lepton_all_mask & ~neutrino_mask

    lepton_pairs = events.GenPart[lepton_mask]
    lepton_pairs_vis = events.GenPart[lepton_vis_mask]
    lepton_pairs_all = events.GenPart[lepton_all_mask]

    dilepton_p4 = _sum_gen_p4(lepton_pairs)
    vis_p4 = _sum_gen_p4(lepton_pairs_vis)
    all_p4 = _sum_gen_p4(lepton_pairs_all)

    events = set_ak_column(
        events,
        "gen_dilepton_pdgid",
        ak.without_parameters(abs(_safe_first(lepton_pairs.pdgId, default=0))),
    )
    events = set_ak_column(
        events,
        "gen_dilepton_pt",
        dilepton_p4.pt,
        value_type=np.float32,
    )

    events = set_ak_column(
        events,
        "gen_dilepton_vis.pt",
        vis_p4.pt,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_vis.eta",
        vis_p4.eta,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_vis.phi",
        vis_p4.phi,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_vis.mass",
        vis_p4.mass,
        value_type=np.float32,
    )

    events = set_ak_column(
        events,
        "gen_dilepton_all.pt",
        all_p4.pt,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_all.eta",
        all_p4.eta,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_all.phi",
        all_p4.phi,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "gen_dilepton_all.mass",
        all_p4.mass,
        value_type=np.float32,
    )

    return events


# correctionlib systematic name -> output column suffix
RECOIL_MET_SYSTS = {
    "RespUp": "recoilresp_up",
    "RespDown": "recoilresp_down",
    "ResolUp": "recoilres_up",
    "ResolDown": "recoilres_down",
}


def _current_shift_name(task: law.Task) -> str:
    """
    Return the active ColumnFlow local shift name using the task instance.

    This avoids deprecated direct access to self.local_shift_inst and
    self.global_shift_inst.
    """

    shift_inst = getattr(task, "local_shift_inst", None)
    if shift_inst is not None:
        return shift_inst.name

    return "nominal"


@producer(
    uses={
        "PuppiMET.{pt,phi,covXX,covXY,covYY}",
        gen_dilepton.PRODUCES,
    },
    produces={
        "RecoilCorrMET.{pt,phi,covXX,covXY,covYY}",
        "RecoilCorrMET.{pt,phi}_{recoilresp,recoilres}_{up,down}",
    },
    njet_column=None,
    mc_only=True,
)
def recoil_corrected_met(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> ak.Array:
    """
    Producer for bosonic recoil corrections.

    Nominal correction:
      U = MET + V_vis - V_full
      U_corr = recoil correction applied to Upara/Uperp
      MET_corr = U_corr - V_vis + V_full

    Uncertainties:
      H = -(MET_corr + V_vis)
      H varied through Recoil_correction_Uncertainty
      MET_var = -H_var - V_vis
    """

    import vector

    dataset_name = self.dataset_inst.name

    recoil_input_met_pt = events.PuppiMET.pt
    recoil_input_met_phi = events.PuppiMET.phi

    met_covXX = events.PuppiMET.covXX
    met_covXY = events.PuppiMET.covXY
    met_covYY = events.PuppiMET.covYY

    # For datasets where recoil corrections are not configured, still create
    # the RecoilCorrMET columns so downstream code can use a common MET branch.
    if dataset_name not in self.met_recoil_datasets:
        events = set_ak_column(
            events,
            "RecoilCorrMET.pt",
            recoil_input_met_pt,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "RecoilCorrMET.phi",
            recoil_input_met_phi,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "RecoilCorrMET.covXX",
            met_covXX,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "RecoilCorrMET.covXY",
            met_covXY,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "RecoilCorrMET.covYY",
            met_covYY,
            value_type=np.float32,
        )

        for postfix in RECOIL_MET_SYSTS.values():
            events = set_ak_column(
                events,
                f"RecoilCorrMET.pt_{postfix}",
                recoil_input_met_pt,
                value_type=np.float32,
            )
            events = set_ak_column(
                events,
                f"RecoilCorrMET.phi_{postfix}",
                recoil_input_met_phi,
                value_type=np.float32,
            )

        return events

    order = self.met_recoil_datasets[dataset_name]

    logger.info(
        "applying MET recoil correction for dataset %s with order=%s",
        dataset_name,
        order,
    )

    # -------------------------------------------------------------------------
    # Build transverse vectors
    # -------------------------------------------------------------------------
    met = vector.array(
        {
            "pt": recoil_input_met_pt,
            "phi": recoil_input_met_phi,
            "eta": np.zeros_like(recoil_input_met_pt),
            "mass": np.zeros_like(recoil_input_met_pt),
        }
    )

    full = vector.array(
        {
            "pt": events.gen_dilepton_all.pt,
            "phi": events.gen_dilepton_all.phi,
            "eta": np.zeros_like(events.gen_dilepton_all.pt),
            "mass": np.zeros_like(events.gen_dilepton_all.pt),
        }
    )

    vis = vector.array(
        {
            "pt": events.gen_dilepton_vis.pt,
            "phi": events.gen_dilepton_vis.phi,
            "eta": np.zeros_like(events.gen_dilepton_vis.pt),
            "mass": np.zeros_like(events.gen_dilepton_vis.pt),
        }
    )

    # -------------------------------------------------------------------------
    # U = MET + visible boson - full boson
    # -------------------------------------------------------------------------
    u_x = met.x + vis.x - full.x
    u_y = met.y + vis.y - full.y

    # -------------------------------------------------------------------------
    # Project U along and perpendicular to the full-boson axis
    # -------------------------------------------------------------------------
    full_pt = full.pt
    safe_full_pt = ak.where(full_pt > 0.0, full_pt, 1.0)

    full_unit_x = full.x / safe_full_pt
    full_unit_y = full.y / safe_full_pt

    upara = u_x * full_unit_x + u_y * full_unit_y
    uperp = -u_x * full_unit_y + u_y * full_unit_x

    # -------------------------------------------------------------------------
    # Jet multiplicity used by the recoil JSON:
    #
    #   pT > 50 GeV within 2.5 < |eta| < 3.0
    #   pT > 30 GeV otherwise
    #
    # The correction uses njet as float and the example clips njet > 2 to 2.
    # -------------------------------------------------------------------------
    if self.njet_column:
        njet = np.asarray(events[self.njet_column], dtype=np.float32)
    else:
        abs_eta = np.abs(events.Jet.eta)
        in_transition_region = (abs_eta > 2.5) & (abs_eta < 3.0)

        jet_selection = ak.where(
            in_transition_region,
            events.Jet.pt > 50.0,
            events.Jet.pt > 30.0,
        )

        selected_jets = events.Jet[jet_selection]
        njet = np.asarray(ak.num(selected_jets, axis=1), dtype=np.float32)

    njet = np.minimum(njet, 2.0).astype(np.float32)

    # Full gen-level boson pT, including neutrinos.
    ptll = np.asarray(events.gen_dilepton_all.pt, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Nominal correction with QuantileMapHist
    #
    # By-era files use:
    #   order, njet, ptll, var, val
    #
    # Global files would instead require:
    #   era, order, njet, ptll, var, val
    # -------------------------------------------------------------------------
    upara_corr = self.recoil_corrector.evaluate(
        order,
        njet,
        ptll,
        "Upara",
        upara,
    )

    uperp_corr = self.recoil_corrector.evaluate(
        order,
        njet,
        ptll,
        "Uperp",
        uperp,
    )

    # Reassemble corrected U vector.
    ucorr_x = upara_corr * full_unit_x - uperp_corr * full_unit_y
    ucorr_y = upara_corr * full_unit_y + uperp_corr * full_unit_x

    # MET_corr = U_corr - visible boson + full boson
    met_corr_x = ucorr_x - vis.x + full.x
    met_corr_y = ucorr_y - vis.y + full.y

    met_corr_pt = np.sqrt(met_corr_x**2 + met_corr_y**2)
    met_corr_phi = np.arctan2(met_corr_y, met_corr_x)

    events = set_ak_column(
        events,
        "RecoilCorrMET.pt",
        met_corr_pt,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "RecoilCorrMET.phi",
        met_corr_phi,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "RecoilCorrMET.covXX",
        met_covXX,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "RecoilCorrMET.covXY",
        met_covXY,
        value_type=np.float32,
    )
    events = set_ak_column(
        events,
        "RecoilCorrMET.covYY",
        met_covYY,
        value_type=np.float32,
    )

    # -------------------------------------------------------------------------
    # Recoil uncertainties
    #
    # The uncertainty correction expects Hpara/Hperp, not Upara/Uperp.
    #
    # H = -(MET_corr + visible boson)
    # -------------------------------------------------------------------------
    h_x = -met_corr_x - vis.x
    h_y = -met_corr_y - vis.y

    hpara = h_x * full_unit_x + h_y * full_unit_y
    hperp = -h_x * full_unit_y + h_y * full_unit_x

    active_shift = _current_shift_name(task)
    recoil_shift_cache = {}

    for syst, postfix in RECOIL_MET_SYSTS.items():
        hpara_var = self.recoil_unc_corrector.evaluate(
            order,
            njet,
            ptll,
            "Hpara",
            hpara,
            syst,
        )

        hperp_var = self.recoil_unc_corrector.evaluate(
            order,
            njet,
            ptll,
            "Hperp",
            hperp,
            syst,
        )

        hcorr_x = hpara_var * full_unit_x - hperp_var * full_unit_y
        hcorr_y = hpara_var * full_unit_y + hperp_var * full_unit_x

        # MET_var = -H_var - visible boson
        met_var_x = -hcorr_x - vis.x
        met_var_y = -hcorr_y - vis.y

        met_var_pt = np.sqrt(met_var_x**2 + met_var_y**2)
        met_var_phi = np.arctan2(met_var_y, met_var_x)

        events = set_ak_column(
            events,
            f"RecoilCorrMET.pt_{postfix}",
            met_var_pt,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            f"RecoilCorrMET.phi_{postfix}",
            met_var_phi,
            value_type=np.float32,
        )

        recoil_shift_cache[postfix] = (met_var_pt, met_var_phi)

    # For an active shifted branch, overwrite the nominal route so downstream
    # producers using events.RecoilCorrMET.pt/phi see the shifted MET.
    if active_shift in recoil_shift_cache:
        active_met_pt, active_met_phi = recoil_shift_cache[active_shift]

        events = set_ak_column(
            events,
            "RecoilCorrMET.pt",
            active_met_pt,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "RecoilCorrMET.phi",
            active_met_phi,
            value_type=np.float32,
        )

    return events


@recoil_corrected_met.init
def recoil_corrected_met_init(self: Producer) -> None:
    if self.njet_column:
        self.uses.add(f"{self.njet_column}")
    else:
        self.uses.add("Jet.{pt,eta,phi,mass}")

    # Declare recoil shifts handled by this producer.
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("met_recoil")
    }


@recoil_corrected_met.requires
def recoil_corrected_met_requires(
    self: Producer,
    task: law.Task,
    reqs: dict,
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles

    reqs["external_files"] = BundleExternalFiles.req(task)


@recoil_corrected_met.setup
def recoil_corrected_met_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    bundle = reqs["external_files"]

    import correctionlib

    correctionlib.highlevel.Correction.__call__ = (
        correctionlib.highlevel.Correction.evaluate
    )

    correction_set = correctionlib.CorrectionSet.from_string(
        bundle.files.met_recoil.load(formatter="gzip").decode("utf-8")
    )

    self.recoil_corrector = correction_set[
        "Recoil_correction_QuantileMapHist"
    ]

    self.recoil_unc_corrector = correction_set[
        "Recoil_correction_Uncertainty"
    ]

    self.met_recoil_datasets = self.config_inst.x.met_recoil.datasets