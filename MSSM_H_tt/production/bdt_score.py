# coding: utf-8

"""
Producer for the MSSM H->tautau e-mu 4-class BDT using the 10-feature
current ggphi/bbphi training.

Training classes / analysis regions:
  0: ggphi_phitautau -> ggphi
  1: bbphi_phitautau -> bbphi
  2: DY              -> dy
  3: TT              -> tt

For each mass point, the producer writes:
  - raw four-class probabilities: bdt_raw_score_{ggphi,bbphi,dy,tt}_M{mass}
  - four-class argmax category: bdt_cat_M{mass}
  - probability discriminants:
      bdt_D_sig_M{mass}
      bdt_D_ggphi_M{mass}
      bdt_D_bbphi_M{mass}
      bdt_Disc_ggphi_M{mass}
      bdt_Disc_bbphi_M{mass}
      bdt_D_DY_M{mass}
      bdt_D_TT_M{mass}
      bdt_D_bbphi_sig_M{mass}
      bdt_D_ggphi_sig_M{mass}

The feature list and order must match the BDT training script:
  n_bjets, delta_eta_jj, mt_tot, fastMTT, pt_lead_b_jet,
  eta_lead_b_jet, eta_sublead_jet, pt_sublead_jet, m_vis, D_zeta

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

# Keep this synchronized with the training script.
CLIP_JET_COUNT_FEATURES = True
N_BJETS_CLIP_MIN = 0.0
N_BJETS_CLIP_MAX = 2.0
JET_COUNT_MODE_TAG = "clippedJetCounts" if CLIP_JET_COUNT_FEATURES else "rawJetCounts"

# Must match OUTPUT_BASE in the BDT training script.
BDT_EOS_BASE = Path(
    "/eos/project/d/desytau/public/jmalvaso/"
    f"bdt_3_classes_10_features_{JET_COUNT_MODE_TAG}"
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
        "No best-D_sig penalty summary found for M=%s in %s. "
        "Falling back to w_DY=w_TT=1.",
        mass,
        BDT_EOS_BASE,
    )
    return 1.0, 1.0


BDT_LABELS = [
    "ggphi",  # 0: ggphi_phitautau
    "bbphi",  # 1: bbphi_phitautau
    "dy",     # 2: DY
    "tt",     # 3: TT
]

# Feature names and order must match the training script. Do not add event,
# event_weight or event_weight_train here; they are not BDT inputs.
BDT_FEATURES = [
    "n_bjets",
    "delta_eta_jj",
    "mt_tot",
    "fastMTT",
    "pt_lead_b_jet",
    "eta_lead_b_jet",
    "eta_sublead_jet",
    "pt_sublead_jet",
    "m_vis",
    "D_zeta",
]

BDT_DISCRIMINANTS = [
    "D_sig",
    "D_ggphi",
    "D_bbphi",
    "Disc_ggphi",
    "Disc_bbphi",
    "D_DY",
    "D_TT",
    "D_bbphi_sig",
    "D_ggphi_sig",
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
            raise KeyError(
                f"missing field '{part}' while resolving '{path}', available={fields}"
            )
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


def _clip_if_enabled_n_bjets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not CLIP_JET_COUNT_FEATURES:
        return values
    return np.clip(values, N_BJETS_CLIP_MIN, N_BJETS_CLIP_MAX).astype(np.float32)


# -------------------------------------------------------------------------
# Feature construction
# -------------------------------------------------------------------------

def _build_bdt_feature_frame(events: ak.Array, channel: str):
    """
    Build the exact feature frame used by the 10-feature BDT training script.
    """
    hcand = events[f"hcand_{channel}"]

    n_bjets = _flat_first_existing(events, ["n_bjets", "N_b_jets"])
    n_bjets = _clip_if_enabled_n_bjets(n_bjets)

    features_dict = {
        "n_bjets": n_bjets,
        "delta_eta_jj": _flat_first_existing(events, ["delta_eta_jj", "dijet.deltaeta"]),
        "mt_tot": _flat_first_existing(events, ["mt_tot", f"hcand_{channel}.mt_tot"]),
        "fastMTT": _flat_first_existing(events, ["fastMTT", f"hcand_{channel}.fastMTT.mass"], axis=1),
        "pt_lead_b_jet": _flat_first_existing(events, ["pt_lead_b_jet", "lead_b_jet.pt"]),
        "eta_lead_b_jet": _flat_first_existing(events, ["eta_lead_b_jet", "lead_b_jet.eta"]),
        "eta_sublead_jet": _flat_first_existing(events, ["eta_sublead_jet", "sublead_jet.eta"]),
        "pt_sublead_jet": _flat_first_existing(events, ["pt_sublead_jet", "sublead_jet.pt"]),
        "m_vis": _flat_first_existing(events, ["m_vis", f"hcand_{channel}.mass"]),
        "D_zeta": _flat_first_existing(events, ["D_zeta"]),
    }

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
            "N_b_jets",
            "dijet.deltaeta",
            "hcand_*.mt_tot",
            "hcand_*.fastMTT.mass",
            "lead_b_jet.pt",
            "lead_b_jet.eta",
            "sublead_jet.pt",
            "sublead_jet.eta",
            "hcand_*.mass",
            "D_zeta",
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
    Return per-mass BDT scores, probability discriminants and four-class categories.
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


    for mass in self.mass_points:
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

        common_signal_den = p_sig + w_DY * p_DY + w_TT * p_TT + eps
        D_ggphi = p_ggphi / common_signal_den
        D_bbphi = p_bbphi / common_signal_den

        discriminants = {
                "D_sig": p_sig / common_signal_den,
                "D_ggphi": D_ggphi,
                "D_bbphi": D_bbphi,
                "Disc_ggphi": D_ggphi / (D_ggphi + D_bbphi + eps),
                "Disc_bbphi": D_bbphi / (D_ggphi + D_bbphi + eps),
                "D_DY": p_DY / (p_sig + p_DY + w_TT * p_TT + eps),
                "D_TT": p_TT / (p_sig + w_DY * p_DY + p_TT + eps),
                "D_bbphi_sig": p_bbphi / (p_bbphi + p_ggphi + eps),
                "D_ggphi_sig": p_ggphi / (p_ggphi + p_bbphi + eps),
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
    from MSSM_H_tt.ml.xgb_evaluator import XGBEvaluator

    self.evaluator = XGBEvaluator()
    self.bdt_best_dsig_penalties = {}

    for mass in self.mass_points:
        self.evaluator.add_model(f"bdt_even_M{mass}", _even_path(mass))
        self.evaluator.add_model(f"bdt_odd_M{mass}", _odd_path(mass))

        self.bdt_best_dsig_penalties[int(mass)] = (
            _read_best_dsig_penalties(int(mass))
        )

    # IMPORTANT: load models only once for the complete ProduceColumns task
    self.evaluator.start()


@mssm_bdt_score.teardown
def mssm_bdt_score_teardown(self: Producer, **kwargs) -> None:
    """
    Stop the XGB evaluator.
    """
    if (evaluator := getattr(self, "evaluator", None)) is not None:
        evaluator.stop()