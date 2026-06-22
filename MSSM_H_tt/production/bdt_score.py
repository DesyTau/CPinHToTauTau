# coding: utf-8

"""
Producer for the MSSM H->tautau e-mu 4-class BDT using the current ggphi/bbphi
training.

Training classes / analysis regions:
  0: ggphi_phitautau -> ggphi
  1: bbphi_phitautau -> bbphi
  2: DY              -> dy
  3: TT              -> tt

For each mass point, the producer writes:
  - raw four-class probabilities: bdt_raw_score_{ggphi,bbphi,dy,tt}_M{mass}
  - four-class argmax category: bdt_cat_M{mass}
  - region discriminants:
      bdt_D_sig_M{mass}
      bdt_D_ggphi_M{mass}
      bdt_D_bbphi_M{mass}
      bdt_D_DY_M{mass}
      bdt_D_TT_M{mass}

The four fit regions use:
  bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)

The per-region discriminants are computed with the w_DY and w_TT values from
best D_sig in the training penalty scan when the summary files are available.
When no summary file is found, the producer falls back to w_DY = w_TT = 1.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import law
from columnflow.util import maybe_import, DotDict, dev_sandbox
from columnflow.columnar_util import set_ak_column, flat_np_view
from columnflow.types import Any
from columnflow.production import Producer, producer

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
pd = maybe_import("pandas")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


# -------------------------------------------------------------------------
# Mass points and model locations
# -------------------------------------------------------------------------
from MSSM_H_tt.config.mass_points import read_bdt_masses
MASS_POINTS = read_bdt_masses()

# Must match OUTPUT_BASE in the BDT training script, i.e. the directory where
# the trained JSON models and fixed-cut penalty summaries are stored.
#
# Current training script uses raw jet-count features by default. If your jobs
# were intentionally written to another output directory, change only this path.
BDT_EOS_BASE = Path(
    "/eos/project/d/desytau/public/jmalvaso/"
    "bdt_4_classes_phi_no_DY_tail_focus_normWeightTraining_clippedJetCounts"
)


def _even_path(mass: int) -> str:
    return str(BDT_EOS_BASE / f"M{int(mass)}" / f"bst_model_M{int(mass)}_even.json")


def _odd_path(mass: int) -> str:
    return str(BDT_EOS_BASE / f"M{int(mass)}" / f"bst_model_M{int(mass)}_odd.json")


def _four_region_summary_path(mass: int) -> Path:
    return (
        BDT_EOS_BASE
        / f"M{int(mass)}"
        / "four_bdt_regions_best_Dsig_penalties"
        / f"four_region_bestDsig_penalty_summary_M{int(mass)}_combined_crossApplied.json"
    )


def _fixedcut_best_path(mass: int) -> Path:
    return (
        BDT_EOS_BASE
        / f"M{int(mass)}"
        / "penalty_scan_combined_crossApplied"
        / "fixed_cut"
        / f"fixedcut_penalty_best_M{int(mass)}_combined_crossApplied.json"
    )


def _read_best_dsig_penalties(mass: int) -> tuple[float, float]:
    """
    Read the best D_sig penalty pair produced by the training scan.

    Preferred source:
      M*/four_bdt_regions_best_Dsig_penalties/four_region_bestDsig_penalty_summary_*.json

    Fallback source:
      M*/penalty_scan_combined_crossApplied/fixed_cut/fixedcut_penalty_best_*.json

    Final fallback:
      w_DY = 1, w_TT = 1
    """
    summary_path = _four_region_summary_path(mass)
    if summary_path.is_file():
        try:
            rows = json.loads(summary_path.read_text())
            if rows:
                row = rows[0]
                return (
                    float(row["dy_penalty_from_best_Dsig"]),
                    float(row["tt_penalty_from_best_Dsig"]),
                )
        except Exception as exc:
            logger.warning(
                "Could not read four-region D_sig penalty summary %s: %s",
                summary_path,
                exc,
            )

    fixed_path = _fixedcut_best_path(mass)
    if fixed_path.is_file():
        try:
            rows = json.loads(fixed_path.read_text())
            for row in rows:
                if str(row.get("discriminant")) == "D_sig":
                    return float(row["dy_penalty"]), float(row["tt_penalty"])
        except Exception as exc:
            logger.warning(
                "Could not read fixed-cut D_sig penalty summary %s: %s",
                fixed_path,
                exc,
            )

    logger.warning(
        "No best-D_sig penalty summary found for M=%s. Falling back to w_DY=w_TT=1.",
        mass,
    )
    return 1.0, 1.0


BDT_LABELS = [
    "ggphi",  # 0: ggphi_phitautau
    "bbphi",  # 1: bbphi_phitautau
    "dy",     # 2: DY
    "tt",     # 3: TT
]

BDT_PNETB_TOP_K = 8


def _jet_raw_pnetb_feature_names() -> list[str]:
    return [f"jet_raw_PNetB_jet{i}" for i in range(1, BDT_PNETB_TOP_K + 1)]


# Feature names and order must match the training script. Do not add event or
# event_weight here; they are not BDT inputs.
BDT_FEATURES = [
    "mt_tot",
    "mt_jets",
    "mt_bjets",
    "mjj",
    "mb_jb_jb",
    "n_jets",
    "n_bjets",
    "fastMTT",
    "delta_eta_jj",
    "delta_eta_bb",
    "pt_lead_jet",
    "pt_sublead_jet",
    "pt_lead_b_jet",
    "eta_lead_jet",
    "eta_sublead_jet",
    "eta_lead_b_jet",
    *_jet_raw_pnetb_feature_names(),
    "D_zeta",
    "pt_e",
    "pt_mu",
    "eta_e",
    "eta_mu",
    "met_pt",
    "dR_emu",
    "m_vis",
    "mt_e",
    "mt_mu",
    "mt_emu",
]

BDT_DISCRIMINANTS = [
    "D_sig",
    "D_ggphi",
    "D_bbphi",
    "D_DY",
    "D_TT",
]


# -------------------------------------------------------------------------
# Feature construction helpers
# -------------------------------------------------------------------------

def _ak_fields(arr) -> list[str]:
    try:
        return ak.fields(arr)
    except Exception:
        return []


def _get_nested(arr, path: str):
    """Resolve a dotted awkward field path."""
    out = arr
    for part in path.split("."):
        fields = _ak_fields(out)
        if part not in fields:
            raise KeyError(f"missing field '{part}' while resolving '{path}', available={fields}")
        out = out[part]
    return out


def _flat_float(x, axis=-1, fill=np.nan):
    x = ak.fill_none(x, fill)
    return np.asarray(flat_np_view(x, axis=axis), dtype=np.float32)


def _flat_first_existing(events: ak.Array, paths: list[str], axis=-1):
    last_exc = None
    for path in paths:
        try:
            return _flat_float(_get_nested(events, path), axis=axis)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not build feature from any of {paths}") from last_exc


def _pt_or_rho(obj):
    """Return transverse momentum from common vector layouts."""
    for name in ("pt", "rho"):
        try:
            return _get_nested(obj, name)
        except Exception:
            pass

    for x_name, y_name in (("px", "py"), ("x", "y"), ("fX", "fY")):
        try:
            x = _get_nested(obj, x_name)
            y = _get_nested(obj, y_name)
            return np.sqrt(x * x + y * y)
        except Exception:
            pass

    raise RuntimeError(f"Could not extract transverse momentum from fields {_ak_fields(obj)}")


def _flat_pt_or_rho(obj, axis=-1):
    return _flat_float(_pt_or_rho(obj), axis=axis)


def _build_jet_raw_pnetb_slots(events: ak.Array) -> dict[str, np.ndarray]:
    """
    Build fixed jet_raw_PNetB_jet1...jet8 columns from the event-level jagged
    per-jet b-tag array.

    Preferred source order:
      1. Jet.btagPNetB
      2. top-level jet_raw_PNetB
    """
    sources = ["Jet.btagPNetB", "jet_raw_PNetB"]
    x = None
    chosen = None

    for source in sources:
        try:
            x = _get_nested(events, source)
            chosen = source
            break
        except Exception:
            pass

    if x is None:
        raise RuntimeError(
            "Could not find per-jet PNetB input. Tried: " + ", ".join(sources)
        )

    logger.debug("Using %s as jet_raw_PNetB input", chosen)

    rows = ak.to_list(x)
    n_events = len(rows)

    def scalar(value):
        if value is None:
            return np.nan
        while isinstance(value, (list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[0]
        try:
            return float(value)
        except Exception:
            return np.nan

    out = {}
    for i in range(BDT_PNETB_TOP_K):
        values = np.full(n_events, np.nan, dtype=np.float32)
        for iev, row in enumerate(rows):
            if row is None or len(row) <= i:
                continue
            values[iev] = scalar(row[i])
        out[f"jet_raw_PNetB_jet{i + 1}"] = values

    return out


# -------------------------------------------------------------------------
# Feature construction
# -------------------------------------------------------------------------

def _build_bdt_feature_frame(events: ak.Array, channel: str):
    """
    Build the exact feature frame used by train_bdt_4class_htcondor.py.
    """
    hcand = events[f"hcand_{channel}"]

    features_dict = {
        "mt_tot": _flat_float(hcand.mt_tot),
        "mt_jets": _flat_first_existing(events, ["mt_jets"]),
        "mt_bjets": _flat_first_existing(events, ["mt_bjets"]),
        "mjj": _flat_first_existing(events, ["mjj", "dijet.mass"]),
        "mb_jb_jb": _flat_first_existing(events, ["mb_jb_jb", "di_b_jet.mass"]),

        # Current training uses raw, unclipped jet-count features.
        "n_jets": _flat_first_existing(events, ["n_jets"]),
        "n_bjets": _flat_first_existing(events, ["n_bjets", "N_b_jets"]),

        "fastMTT": _flat_float(hcand.fastMTT.mass, axis=1),
        "delta_eta_jj": _flat_first_existing(events, ["delta_eta_jj", "dijet.deltaeta"]),
        "delta_eta_bb": _flat_first_existing(events, ["delta_eta_bb", "di_b_jet.deltaeta"]),
        "pt_lead_jet": _flat_first_existing(events, ["pt_lead_jet", "lead_jet.pt"]),
        "pt_sublead_jet": _flat_first_existing(events, ["pt_sublead_jet", "sublead_jet.pt"]),
        "pt_lead_b_jet": _flat_first_existing(events, ["pt_lead_b_jet", "lead_b_jet.pt"]),
        "eta_lead_jet": _flat_first_existing(events, ["eta_lead_jet", "lead_jet.eta"]),
        "eta_sublead_jet": _flat_first_existing(events, ["eta_sublead_jet", "sublead_jet.eta"]),
        "eta_lead_b_jet": _flat_first_existing(events, ["eta_lead_b_jet", "lead_b_jet.eta"]),

        "D_zeta": _flat_first_existing(events, ["D_zeta"]),
        "pt_e": _flat_float(hcand.lep0.pt, axis=1),
        "pt_mu": _flat_float(hcand.lep1.pt, axis=1),
        "eta_e": _flat_float(hcand.lep0.eta, axis=1),
        "eta_mu": _flat_float(hcand.lep1.eta, axis=1),

        "met_pt": _flat_first_existing(
            events,
            [
                "met_pt",
                "PuppiMET.pt",
                "PuppiMET.rho",
                "RecoilCorrMET.pt",
                "RecoilCorrMET.rho",
            ],
        ),

        "dR_emu": _flat_float(hcand.delta_r),
        "m_vis": _flat_float(hcand.mass),
        "mt_e": _flat_float(hcand.mt_e),
        "mt_mu": _flat_float(hcand.mt_mu),
        "mt_emu": _flat_float(hcand.mt_emu),
    }

    features_dict.update(_build_jet_raw_pnetb_slots(events))

    features = pd.DataFrame.from_dict(features_dict)
    features = features.replace([np.inf, -np.inf], np.nan)

    missing = [f for f in BDT_FEATURES if f not in features.columns]
    if missing:
        raise RuntimeError("Missing BDT input features: " + ", ".join(missing))

    return features[BDT_FEATURES]


def _eval_model_or_empty(evaluator, key, features, n_classes):
    if len(features) == 0:
        return np.empty((0, n_classes), dtype=np.float32)

    res = np.asarray(evaluator(key, features), dtype=np.float32)

    if res.ndim == 1:
        res = res.reshape(-1, n_classes)

    return res[:, :n_classes]


# -------------------------------------------------------------------------
# Producer
# -------------------------------------------------------------------------
@producer(
    uses={
        "event",
        "hcand_*.*",
        "hcand_*.fastMTT.*",
        "lead_jet.pt", "lead_jet.eta", "lead_jet.phi",
        "sublead_jet.pt", "sublead_jet.eta", "sublead_jet.phi",
        "lead_b_jet.pt", "lead_b_jet.eta", "lead_b_jet.phi",
        "sublead_b_jet.pt", "sublead_b_jet.phi",
        "di_b_jet.mass", "di_b_jet.deltaeta",
        "dijet.mass", "dijet.deltaeta",
        "mt_jets", "mt_bjets",
        "n_jets", "N_b_jets",
        "D_zeta",
        "PuppiMET.*",
        "RecoilCorrMET.*",
        "Jet.btagPNetB",
    },
    produces=(
        {
            f"bdt_raw_score_{lbl}_M{m}"
            for m in MASS_POINTS
            for lbl in BDT_LABELS
        }
        | {
            f"bdt_{disc}_M{m}"
            for m in MASS_POINTS
            for disc in BDT_DISCRIMINANTS
        }
        | {f"bdt_cat_M{m}" for m in MASS_POINTS}
    ),
    sandbox=dev_sandbox("bash::$HTTCP_BASE/sandboxes/venv_columnar_xgb.sh"),
)
def mssm_bdt_score(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Return per-mass BDT scores, region discriminants and four-class categories.
    """
    channel = self.config_inst.channels.names()[0]
    event_n = flat_np_view(events.event)

    features = _build_bdt_feature_frame(events, channel)
    features.index = event_n

    mask_even = (event_n % 2 == 0)
    features_even = features.loc[mask_even]
    features_odd = features.loc[~mask_even]

    n_classes = len(BDT_LABELS)
    eps = np.float32(1e-12)

    with self.evaluator:
        for mass in MASS_POINTS:
            key_even = f"bdt_even_M{mass}"
            key_odd = f"bdt_odd_M{mass}"

            # Cross-apply the parity models, matching the training script:
            #   - even model was trained on even events and is applied to odd events
            #   - odd model was trained on odd events and is applied to even events
            res_even_model_on_odd = _eval_model_or_empty(
                self.evaluator,
                key_even,
                features_odd,
                n_classes,
            )
            res_odd_model_on_even = _eval_model_or_empty(
                self.evaluator,
                key_odd,
                features_even,
                n_classes,
            )

            output = np.zeros((len(event_n), n_classes), dtype=np.float32)
            output[mask_even, :] = res_odd_model_on_even
            output[~mask_even, :] = res_even_model_on_odd

            p_ggphi = output[:, 0]
            p_bbphi = output[:, 1]
            p_DY = output[:, 2]
            p_TT = output[:, 3]
            p_sig = p_ggphi + p_bbphi

            # Four analysis regions: argmax(P_ggphi, P_bbphi, P_DY, P_TT).
            bdt_cat = np.argmax(output, axis=1).astype(np.int32)

            dy_penalty, tt_penalty = self.bdt_best_dsig_penalties.get(
                int(mass),
                (1.0, 1.0),
            )
            w_DY = np.float32(dy_penalty)
            w_TT = np.float32(tt_penalty)

            common_signal_den = p_ggphi + p_bbphi + w_DY * p_DY + w_TT * p_TT + eps
            discriminants = {
                "D_sig": p_sig / common_signal_den,
                "D_ggphi": p_ggphi / common_signal_den,
                "D_bbphi": p_bbphi / common_signal_den,
                "D_DY": p_DY / (p_sig + p_DY + w_TT * p_TT + eps),
                "D_TT": p_TT / (p_sig + w_DY * p_DY + p_TT + eps),
            }

            for idx, label in enumerate(BDT_LABELS):
                events = set_ak_column_f32(
                    events,
                    f"bdt_raw_score_{label}_M{mass}",
                    np.ascontiguousarray(output[:, idx]),
                )

            for name, values in discriminants.items():
                events = set_ak_column_f32(
                    events,
                    f"bdt_{name}_M{mass}",
                    np.ascontiguousarray(values.astype(np.float32)),
                )

            events = set_ak_column_i32(
                events,
                f"bdt_cat_M{mass}",
                np.ascontiguousarray(bdt_cat),
            )

    return events


@mssm_bdt_score.requires
def mssm_bdt_score_requires(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    """
    No external bundle needed; models and penalty summaries are read directly
    from EOS in setup.
    """
    return


@mssm_bdt_score.setup
def mssm_bdt_score_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    """
    Load one XGBoost model pair for each mass point and the best D_sig penalty
    pair used for the region discriminants.
    """
    from MSSM_H_tt.ml.xgb_evaluator import XGBEvaluator

    self.evaluator = XGBEvaluator()
    self.bdt_best_dsig_penalties = {}

    for mass in MASS_POINTS:
        self.evaluator.add_model(f"bdt_even_M{mass}", _even_path(mass))
        self.evaluator.add_model(f"bdt_odd_M{mass}", _odd_path(mass))
        self.bdt_best_dsig_penalties[int(mass)] = _read_best_dsig_penalties(int(mass))


@mssm_bdt_score.teardown
def mssm_bdt_score_teardown(self: Producer, **kwargs) -> None:
    """
    Stop the XGB evaluator.
    """
    if (evaluator := getattr(self, "evaluator", None)) is not None:
        evaluator.stop()