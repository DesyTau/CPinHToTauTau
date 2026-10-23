# coding: utf-8

from __future__ import annotations

import functools
import json
from typing import Optional

import law

from columnflow.columnar_util import (
    optional_column as optional,
    set_ak_column,
)
from columnflow.production import Producer, producer
from columnflow.types import Any
from columnflow.util import DotDict, maybe_import
from law.util import InsertableDict

ak = maybe_import("awkward")
np = maybe_import("numpy")
cl = maybe_import("correctionlib")
schemav2 = maybe_import("correctionlib.schemav2")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default fixed working point. The uploaded 2024 Summer24 UParTAK4 payload gives
# M = 0.1272. The actual threshold is read from UParTAK4_wp_values at runtime,
# so it remains synchronized with the SF payload.
DEFAULT_WORKING_POINT = "M"

# Output suffix, b/c SF variation, light-flavour SF variation.
#
# The simple up/down variations are useful when analysing only one data-taking
# era. For a multi-era combination, use the correlated/uncorrelated breakdown
# instead and decorrelate the uncorrelated nuisance between eras in the config.
FIXED_WP_VARIATIONS = (
    ("nom", "central", "central"),

    # Simple one-era uncertainty scheme
    ("bc_up", "up", "central"),
    ("bc_down", "down", "central"),
    ("light_up", "central", "up"),
    ("light_down", "central", "down"),

    # Multi-era uncertainty scheme
    ("bc_correlated_up", "up_correlated", "central"),
    ("bc_correlated_down", "down_correlated", "central"),
    ("bc_uncorrelated_up", "up_uncorrelated", "central"),
    ("bc_uncorrelated_down", "down_uncorrelated", "central"),
    ("light_correlated_up", "central", "up_correlated"),
    ("light_correlated_down", "central", "down_correlated"),
    ("light_uncorrelated_up", "central", "up_uncorrelated"),
    ("light_uncorrelated_down", "central", "down_uncorrelated"),
)

FIXED_WP_SUFFIXES = tuple(v[0] for v in FIXED_WP_VARIATIONS)


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


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
    Evaluate the MC fixed-WP tagging efficiencies and return a per-jet array
    with the same jagged structure as pt/eta/flavor.

    Hadron-flavour mapping:
      5 -> eff_b
      4 -> eff_c
      0 -> eff_light

    The efficiency maps MUST have been derived for the same jet selection,
    UParTAK4B discriminator, and working point used below.
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


def _evaluate_fixed_wp_sfs(
    self: Producer,
    pt: ak.Array,
    abseta: ak.Array,
    flavor: ak.Array,
    bc_systematic: str,
    light_systematic: str,
) -> ak.Array:
    """
    Evaluate UParTAK4 fixed-WP SFs with the BTV flavour split:

      * b/c jets (hadronFlavour 5/4): UParTAK4_comb
      * light jets (hadronFlavour 0): UParTAK4_light

    The uploaded payload has signature
      (systematic, working_point, flavor, abseta, pt).
    """
    counts = ak.num(pt, axis=1)

    pt_flat = ak.to_numpy(ak.flatten(pt, axis=None)).astype(np.float64)
    abseta_flat = ak.to_numpy(ak.flatten(abseta, axis=None)).astype(np.float64)
    flavor_flat = ak.to_numpy(ak.flatten(flavor, axis=None)).astype(np.int32)

    if len(pt_flat) == 0:
        return ak.unflatten(np.array([], dtype=np.float32), counts)

    sf_flat = np.ones(len(pt_flat), dtype=np.float32)

    mask_bc = (flavor_flat == 5) | (flavor_flat == 4)
    mask_light = ~mask_bc

    if np.any(mask_bc):
        sf_flat[mask_bc] = self.btag_sf_bc_corr.evaluate(
            bc_systematic,
            self.btag_working_point,
            flavor_flat[mask_bc],
            abseta_flat[mask_bc],
            pt_flat[mask_bc],
        )

    if np.any(mask_light):
        # NanoAOD hadronFlavour for light jets is 0. Force the payload input to
        # 0 here so that any unexpected non-4/5 value follows the light branch
        # consistently with the efficiency-map treatment above.
        light_flavor = np.zeros(np.count_nonzero(mask_light), dtype=np.int32)
        sf_flat[mask_light] = self.btag_sf_light_corr.evaluate(
            light_systematic,
            self.btag_working_point,
            light_flavor,
            abseta_flat[mask_light],
            pt_flat[mask_light],
        )

    return ak.unflatten(sf_flat.astype(np.float32), counts)


def _method1a_event_weight(
    sf: ak.Array,
    eff: ak.Array,
    tagged: ak.Array,
) -> ak.Array:
    r"""
    BTV fixed-WP Method 1a event weight:

      tagged jet:       SF
      untagged jet:     (1 - SF * eff) / (1 - eff)

    The event weight is the product over all jets for which the tagging status
    is used in the analysis.
    """
    sf = ak.fill_none(sf, 1.0)
    eff = ak.fill_none(eff, 0.0)
    tagged = ak.fill_none(tagged, False)

    denominator = 1.0 - eff
    untagged_weight = ak.where(
        abs(denominator) > 1.0e-6,
        (1.0 - sf * eff) / denominator,
        1.0,
    )

    per_jet_weight = ak.where(tagged, sf, untagged_weight)
    return ak.prod(per_jet_weight, axis=1)


@producer(
    uses={
        *{f"Jet.{var}" for var in [
            "pt",
            "eta",
            "phi",
            "mass",
            "btagUParTAK4B",
            "pass_tightID_lep_veto",
            "hadronFlavour",
        ]},
        "event",
    },
    produces={
        "btag_weight",
        *(
            optional(f"btag_weight_{suffix}")
            for suffix in FIXED_WP_SUFFIXES
            if suffix != "nom"
        ),
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
    """
    Fixed-WP UParTAK4B b-tagging SF producer using BTV Method 1a.

    IMPORTANT:
      * all analysis jets enter the Method 1a product;
      * passing/failing the WP only decides which per-jet factor is used;
      * b/c and light-flavour SF uncertainties are varied separately.
    """

    # ------------------------------------------------------------------
    # Select every jet for which the b-tagging decision is used.
    # Do NOT require the jet to pass the b-tag WP here: untagged jets are
    # required by Method 1a as well.
    # ------------------------------------------------------------------
    discr_all = events.Jet.btagUParTAK4B
    jet_non_nan_mask = ~np.isnan(discr_all)
    jets = events.Jet[jet_non_nan_mask]

    jet_obj_mask = (
        (jets.pt > 20.0)
        & (abs(jets.eta) < 2.5)
        & jets.pass_tightID_lep_veto
    )
    jets = jets[jet_obj_mask]

    flavor = jets.hadronFlavour
    abseta = abs(jets.eta)
    pt = jets.pt
    discr = jets.btagUParTAK4B

    # Tagging decision for the fixed WP. The threshold comes directly from the
    # UParTAK4_wp_values correction in the same BTV payload.
    tagged = discr >= self.btag_wp_threshold

    # MC efficiencies for the same fixed WP and jet selection.
    eff = _evaluate_btag_efficiencies(self, pt, abseta, flavor)
    events = set_ak_column_f32(events, "btag_eff_selected", eff)

    variations = FIXED_WP_VARIATIONS if do_syst else (FIXED_WP_VARIATIONS[0],)

    for suffix, bc_systematic, light_systematic in variations:
        sf = _evaluate_fixed_wp_sfs(
            self,
            pt,
            abseta,
            flavor,
            bc_systematic=bc_systematic,
            light_systematic=light_systematic,
        )

        w_event = _method1a_event_weight(sf, eff, tagged)

        if suffix == "nom":
            events = set_ak_column_f32(events, "btag_weight", w_event)
        else:
            events = set_ak_column_f32(events, f"btag_weight_{suffix}", w_event)

    return events


@btag_weight_SF.requires
def btag_weight_SF_requires(
    self: Producer,
    task: law.Task,
    reqs: dict,
    **kwargs,
) -> None:
    # No MergeSelectionStats dependency is needed for Method 1a. The b-tagging
    # correction is allowed to change event yields and must not be normalized
    # back to the uncorrected prediction.
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
    bundle = reqs["external_files"]

    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate

    # ------------------------------------------------------------------
    # BTV SF correction set (gzipped JSON)
    # ------------------------------------------------------------------
    sf_cset = _load_correction_set(
        bundle.files.btag_sf_corr,
        formatter="gzip",
    )

    # Optional config override. This lets the same producer be reused if a
    # future payload changes correction names or if a different fixed WP is
    # desired.
    btag_cfg = {}
    if self.config_inst.has_aux("btag_sf_upart"):
        try:
            btag_cfg = dict(self.config_inst.x.btag_sf_upart)
        except Exception:
            btag_cfg = self.config_inst.x.btag_sf_upart

    bc_correction_name = btag_cfg.get("bc_correction_set", "UParTAK4_comb")
    light_correction_name = btag_cfg.get("light_correction_set", "UParTAK4_light")
    wp_correction_name = btag_cfg.get("wp_correction_set", "UParTAK4_wp_values")
    self.btag_working_point = str(
        btag_cfg.get("working_point", DEFAULT_WORKING_POINT)
    ).upper()

    self.btag_sf_bc_corr = sf_cset[bc_correction_name]
    self.btag_sf_light_corr = sf_cset[light_correction_name]
    self.btag_wp_corr = sf_cset[wp_correction_name]

    # Read the numerical discriminator threshold from the same payload instead
    # of duplicating it in Python/config. For the supplied Summer24 payload,
    # working point M evaluates to 0.1272.
    self.btag_wp_threshold = float(
        self.btag_wp_corr.evaluate(self.btag_working_point)
    )

    # ------------------------------------------------------------------
    # MC b-tagging efficiency maps
    # ------------------------------------------------------------------
    # These maps must correspond to UParTAK4B and to the same working point and
    # jet selection used in btag_weight_SF above.
    eff_cset = _load_correction_set(bundle.files.btag_eff_corr)

    self.btag_eff_corr = {
        "eff_b": eff_cset["eff_b"],
        "eff_c": eff_cset["eff_c"],
        "eff_light": eff_cset["eff_light"],
    }