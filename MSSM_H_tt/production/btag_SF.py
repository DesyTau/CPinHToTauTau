# coding: utf-8

from __future__ import annotations

import functools
import json
from typing import Optional

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, DotDict
from law.util import InsertableDict
from columnflow.columnar_util import (
    sorted_indices_from_mask,
    set_ak_column,
    has_ak_column,
    EMPTY_FLOAT,
    Route,
    flat_np_view,
    optional_column as optional,
)
from columnflow.production.util import attach_coffea_behavior
from columnflow.types import Any
import law

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
cl = maybe_import("correctionlib")
schemav2 = maybe_import("correctionlib.schemav2")
warn = maybe_import("warnings")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# deepJet_shape (v1) systematics list
JET_SHAPE_SYSTEMATICS = (
    "central",
    "down_cferr1", "down_cferr2", "down_hf", "down_hfstats1", "down_hfstats2", "down_jes",
    "down_jesAbsoluteMPFBias", "down_jesAbsoluteScale", "down_jesAbsoluteStat",
    "down_jesFlavorQCD", "down_jesFragmentation", "down_jesPileUpDataMC", "down_jesPileUpPtBB",
    "down_jesPileUpPtEC1", "down_jesPileUpPtEC2", "down_jesPileUpPtHF", "down_jesPileUpPtRef",
    "down_jesRelativeBal", "down_jesRelativeFSR", "down_jesRelativeJEREC1", "down_jesRelativeJEREC2",
    "down_jesRelativeJERHF", "down_jesRelativePtBB", "down_jesRelativePtEC1", "down_jesRelativePtEC2",
    "down_jesRelativePtHF", "down_jesRelativeSample", "down_jesRelativeStatEC",
    "down_jesRelativeStatFSR", "down_jesRelativeStatHF", "down_jesSinglePionECAL",
    "down_jesSinglePionHCAL", "down_jesTimePtEta", "down_lf", "down_lfstats1", "down_lfstats2",
    "up_cferr1", "up_cferr2", "up_hf", "up_hfstats1", "up_hfstats2", "up_jes",
    "up_jesAbsoluteMPFBias", "up_jesAbsoluteScale", "up_jesAbsoluteStat",
    "up_jesFlavorQCD", "up_jesFragmentation", "up_jesPileUpDataMC", "up_jesPileUpPtBB",
    "up_jesPileUpPtEC1", "up_jesPileUpPtEC2", "up_jesPileUpPtHF", "up_jesPileUpPtRef",
    "up_jesRelativeBal", "up_jesRelativeFSR", "up_jesRelativeJEREC1", "up_jesRelativeJEREC2",
    "up_jesRelativeJERHF", "up_jesRelativePtBB", "up_jesRelativePtEC1", "up_jesRelativePtEC2",
    "up_jesRelativePtHF", "up_jesRelativeSample", "up_jesRelativeStatEC",
    "up_jesRelativeStatFSR", "up_jesRelativeStatHF", "up_jesSinglePionECAL",
    "up_jesSinglePionHCAL", "up_jesTimePtEta", "up_lf", "up_lfstats1", "up_lfstats2",
)


def _syst_to_suffix(syst: str) -> str:
    if syst == "central":
        return "nom"
    if syst.startswith("up_"):
        return f"{syst[3:]}_up"
    if syst.startswith("down_"):
        return f"{syst[5:]}_down"
    return syst


DEEPJET_SHAPE_SUFFIXES = tuple(_syst_to_suffix(s) for s in JET_SHAPE_SYSTEMATICS)


def _load_correction_set(target, formatter: Optional[str] = None):
    payload = target.load(formatter=formatter) if formatter else target.load()

    # target can return bytes, str, dict, or already a schema object
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    if isinstance(payload, str):
        try:
            return cl.CorrectionSet.from_string(payload)
        except Exception:
            payload = json.loads(payload)

    if isinstance(payload, dict):
        try:
            schema_obj = schemav2.CorrectionSet.model_validate(payload)
        except AttributeError:
            schema_obj = schemav2.CorrectionSet.parse_obj(payload)
        return cl.CorrectionSet(schema_obj)

    return cl.CorrectionSet(payload)


def _evaluate_btag_efficiencies(
    self: Producer,
    pt: ak.Array,
    abseta: ak.Array,
    flavor: ak.Array,
) -> ak.Array:
    """
    Evaluate the WP efficiency maps from btag_eff_corr and return
    a per-jet efficiency array with the same jagged structure as pt/eta/flavor.

    flavor mapping:
      5 -> eff_b
      4 -> eff_c
      else -> eff_light
    """
    counts = ak.num(pt, axis=1)

    pt_flat = ak.to_numpy(ak.flatten(pt, axis=None))
    abseta_flat = ak.to_numpy(ak.flatten(abseta, axis=None))
    flavor_flat = ak.to_numpy(ak.flatten(flavor, axis=None))

    if len(pt_flat) == 0:
        return ak.unflatten(np.array([], dtype=np.float32), counts)

    eff_flat = np.ones(len(pt_flat), dtype=np.float32)

    mask_b = flavor_flat == 5
    mask_c = flavor_flat == 4
    mask_l = ~(mask_b | mask_c)

    if np.any(mask_b):
        eff_flat[mask_b] = self.btag_eff_corr["eff_b"].evaluate(
            pt_flat[mask_b],
            abseta_flat[mask_b],
        )

    if np.any(mask_c):
        eff_flat[mask_c] = self.btag_eff_corr["eff_c"].evaluate(
            pt_flat[mask_c],
            abseta_flat[mask_c],
        )

    if np.any(mask_l):
        eff_flat[mask_l] = self.btag_eff_corr["eff_light"].evaluate(
            pt_flat[mask_l],
            abseta_flat[mask_l],
        )

    return ak.unflatten(eff_flat.astype(np.float32), counts)


@producer(
    uses={
        *{f"Jet.{var}" for var in [
            "pt", "eta", "phi", "mass",
            "btagPNetB", "pass_tightID_lep_veto", "hadronFlavour",
        ]},
        "event",
    },
    produces={
        "btag_weight",
        *(optional(f"btag_weight_{sfx}") for sfx in DEEPJET_SHAPE_SUFFIXES if sfx != "nom"),
        optional("btag_eff_selected"),
    },
    mc_only=True,
)
def btag_weight_SF(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    do_syst: bool,
    **kwargs,
) -> ak.Array:

    systs = JET_SHAPE_SYSTEMATICS if do_syst else ("central",)

    year = self.config_inst.x.year
    tag = self.config_inst.x.tag
    btag_wp = self.config_inst.x.btag_working_points[year][tag].particleNet.medium

    # remove jets with NaN discriminant
    dis_all = events.Jet.btagPNetB
    jet_non_nan_mask = ~np.isnan(dis_all)
    Jet = events.Jet[jet_non_nan_mask]

    # base selection
    jet_obj_mask = (
        (Jet.pt > 20.0)
        & (abs(Jet.eta) < 2.5)
        & (Jet.pass_tightID_lep_veto)
        & (Jet.btagPNetB >= btag_wp)
    )
    Jet = Jet[jet_obj_mask]

    flavor = Jet.hadronFlavour
    abseta = abs(Jet.eta)
    pt = Jet.pt
    discr = Jet.btagPNetB

    # evaluate efficiencies from btag_eff_corr
    eff = _evaluate_btag_efficiencies(self, pt, abseta, flavor)
    events = set_ak_column_f32(events, "btag_eff_selected", eff)

    # central SF
    sf_central = self.btag_sf_corr.evaluate("central", flavor, abseta, pt, discr)
    sf_central = ak.fill_none(sf_central, 1.0)

    # global normalization factor r
    r = 1.0
    try:
        pid = str(self.dataset_inst.processes.ids()[0])
        before = self._selection_stats["sum_mc_weight_selected_per_process"][pid]
        after = self._selection_stats["sum_mc_weight_selected_with_btag_weight_per_process"][pid]
        if after:
            r = before / after
    except Exception:
        r = 1.0
    for syst in systs:
        if syst == "central":
            sf = ak.where(flavor == 5, sf_central, ((1 - sf_central*eff) / (1 - eff))) 
            sf = sf_central

        elif "cferr" in syst:
            flavor_eval = ak.full_like(flavor, 4)
            sf_var = self.btag_sf_corr.evaluate(syst, flavor_eval, abseta, pt, discr)
            sf_var = ak.fill_none(sf_var, 1.0)
            sf = ak.where(flavor == 4, sf_var, sf_central)

        else:
            flavor_eval = ak.where(flavor == 4, 5, flavor)
            sf_var = self.btag_sf_corr.evaluate(syst, flavor_eval, abseta, pt, discr)
            sf_var = ak.fill_none(sf_var, 1.0)
            sf = ak.where(flavor != 4, sf_var, sf_central)

        w_event = ak.prod(sf, axis=1)
        suffix = _syst_to_suffix(syst)

        if suffix == "nom":
            events = set_ak_column_f32(events, "btag_weight", w_event * r)
        else:
            events = set_ak_column_f32(events, f"btag_weight_{suffix}", w_event * r)

    return events


@btag_weight_SF.requires
def btag_weight_SF_requires(
    self: Producer,
    task: law.Task,
    reqs: dict,
    **kwargs,
) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )

    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@btag_weight_SF.setup
def btag_weight_SF_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: InsertableDict,
    **kwargs,
) -> None:
    self._selection_stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    bundle = reqs["external_files"]

    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate

    # SF correction set (gzipped json)
    sf_cset = _load_correction_set(
        bundle.files.btag_sf_corr,
        formatter="gzip",
    )

    if self.config_inst.x.year >= 2022:
        self.btag_sf_corr = sf_cset[self.config_inst.x.btag_sf_pnet.correction_set]
    else:
        self.btag_sf_corr = sf_cset[self.config_inst.x.btag_sf_deepjet.correction_set]

    # efficiency correction set (plain json; target may return dict directly)
    eff_cset = _load_correction_set(bundle.files.btag_eff_corr)

    self.btag_eff_corr = {
        "eff_b": eff_cset["eff_b"],
        "eff_c": eff_cset["eff_c"],
        "eff_light": eff_cset["eff_light"],
    }