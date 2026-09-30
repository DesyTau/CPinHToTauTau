# coding: utf-8

"""
Jet energy corrections / resolution smearing and propagation to PuppiMET.

This implementation keeps JEC and JER as helper calibrators acting on jets,
while a single public calibrator (`jme`) performs the MET propagation using
the NanoAOD Type-1 recipe.

JEC/JER execution is shift-aware:
- nominal jobs compute only nominal JEC/JER,
- a JEC-shift job computes only the requested JEC source/direction and
  propagates nominal JER to that JEC-varied jet branch,
- a JER-shift job computes only the requested JER direction.

The full set of possible output columns is still declared in the init hooks so
that ColumnFlow can build the complete dependency graph.
"""

from __future__ import annotations

import difflib
import functools

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.util import ak_random, sum_transverse
from MSSM_H_tt.calibration.util import propagate_met
from columnflow.util import UNSET, maybe_import, DotDict, load_correction_set
from columnflow.columnar_util import (
    set_ak_column,
    layout_ak_array,
    optional_column as optional,
    ak_concatenate_safe,
)
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")
correctionlib = maybe_import("correctionlib")

logger = law.logger.get_logger(__name__)


#
# helper functions
#

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


def get_active_jme_shift(task):
    """
    Resolve the local calibration shift into the JME variation that actually
    needs to be evaluated.

    Returns
    -------
    tuple[str, str | None, str | None]
        (kind, source, direction), where kind is one of
        "nominal", "jec", or "jer".

    Examples
    --------
    nominal
        -> ("nominal", None, None)

    jec_Regrouped_BBEC1_up
        -> ("jec", "Regrouped_BBEC1", "up")

    jer_down
        -> ("jer", None, "down")
    """
    if task is None:
        return "nominal", None, None

    shift_inst = getattr(task, "local_shift_inst", None)
    if shift_inst is None or shift_inst.name == "nominal":
        return "nominal", None, None

    shift_name = shift_inst.name

    if shift_inst.has_tag("jec"):
        source = shift_inst.x("jec_source")

        if shift_name.endswith("_up"):
            direction = "up"
        elif shift_name.endswith("_down"):
            direction = "down"
        else:
            raise ValueError(
                f"cannot determine JEC direction from local shift '{shift_name}'"
            )

        return "jec", source, direction

    if shift_inst.has_tag("jer"):
        if shift_name == "jer_up":
            return "jer", None, "up"
        if shift_name == "jer_down":
            return "jer", None, "down"

        raise ValueError(
            f"cannot determine JER direction from local shift '{shift_name}'"
        )

    # Unrelated calibration shifts use nominal JME.
    return "nominal", None, None


def _debug_enabled(obj: Any) -> bool:
    return bool(getattr(obj, "debug", False))


def _debug_name(obj: Any) -> str:
    return getattr(obj, "name", obj.__class__.__name__)


def _debug_log(obj: Any, msg: str) -> None:
    if _debug_enabled(obj):
        print(f"[JME-DEBUG:{_debug_name(obj)}] {msg}", flush=True)


def _to_numpy_flat(arr: Any) -> np.ndarray:
    if isinstance(arr, ak.Array):
        try:
            return np.asarray(
                ak.to_numpy(
                    ak.flatten(
                        ak.fill_none(arr, np.nan),
                        axis=None,
                    )
                )
            )
        except Exception:
            return np.asarray(ak.to_list(ak.flatten(arr, axis=None)), dtype=object)
    return np.asarray(arr).reshape(-1)


def _to_numpy_1d(arr: Any) -> np.ndarray:
    if isinstance(arr, ak.Array):
        try:
            return np.asarray(ak.to_numpy(ak.fill_none(arr, np.nan)))
        except Exception:
            return np.asarray(ak.to_list(arr), dtype=object)
    return np.asarray(arr)


def _debug_stats(obj: Any, label: str, arr: Any, max_entries: int | None = None) -> None:
    if not _debug_enabled(obj):
        return

    max_entries = getattr(obj, "debug_max_events", 5) if max_entries is None else max_entries

    try:
        flat = _to_numpy_flat(arr)
    except Exception as e:
        _debug_log(obj, f"{label}: failed to convert array for debug: {e}")
        _debug_log(obj, f"{label}: repr={repr(arr)[:500]}")
        return

    _debug_log(obj, f"{label}: size={flat.size}, dtype={flat.dtype}")

    if flat.size == 0:
        return

    if np.issubdtype(flat.dtype, np.number):
        finite = np.isfinite(flat)
        n_finite = int(np.sum(finite))
        if n_finite > 0:
            vals = flat[finite].astype(np.float64, copy=False)
            _debug_log(
                obj,
                (
                    f"{label}: min={vals.min():.6g}, max={vals.max():.6g}, "
                    f"mean={vals.mean():.6g}, std={vals.std():.6g}"
                ),
            )
        else:
            _debug_log(obj, f"{label}: no finite entries")

    _debug_log(
        obj,
        f"{label}: first {min(max_entries, flat.size)} values = {flat[:max_entries]}",
    )


def _debug_mask(obj: Any, label: str, mask: ak.Array, max_entries: int | None = None) -> None:
    if not _debug_enabled(obj):
        return

    max_entries = getattr(obj, "debug_max_events", 5) if max_entries is None else max_entries

    mask_int = ak.values_astype(mask, np.int64)
    selected = int(ak.sum(mask_int, axis=None))
    total = int(ak.count(mask, axis=None))
    per_event = np.asarray(ak.to_numpy(ak.sum(mask_int, axis=-1)))

    _debug_log(obj, f"{label}: selected {selected} / {total}")
    _debug_log(
        obj,
        f"{label}: first {min(max_entries, len(per_event))} per-event counts = {per_event[:max_entries]}",
    )


def _delta_phi_np(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def _debug_compare_met(
    obj: Any,
    label: str,
    ref_pt: Any,
    ref_phi: Any,
    test_pt: Any,
    test_phi: Any,
    rel_tol: float | None = None,
    max_entries: int | None = None,
) -> None:
    if not _debug_enabled(obj):
        return

    rel_tol = getattr(obj, "debug_rel_tol", 0.01) if rel_tol is None else rel_tol
    max_entries = getattr(obj, "debug_max_events", 5) if max_entries is None else max_entries

    ref_pt_np = _to_numpy_1d(ref_pt).astype(np.float64, copy=False)
    ref_phi_np = _to_numpy_1d(ref_phi).astype(np.float64, copy=False)
    test_pt_np = _to_numpy_1d(test_pt).astype(np.float64, copy=False)
    test_phi_np = _to_numpy_1d(test_phi).astype(np.float64, copy=False)

    if not (len(ref_pt_np) == len(ref_phi_np) == len(test_pt_np) == len(test_phi_np)):
        _debug_log(
            obj,
            (
                f"{label}: length mismatch: "
                f"ref_pt={len(ref_pt_np)}, ref_phi={len(ref_phi_np)}, "
                f"test_pt={len(test_pt_np)}, test_phi={len(test_phi_np)}"
            ),
        )
        return

    if len(ref_pt_np) == 0:
        _debug_log(obj, f"{label}: empty arrays")
        return

    denom = np.maximum(np.abs(ref_pt_np), 1e-6)
    rel_pt = np.abs(test_pt_np - ref_pt_np) / denom
    dphi = np.abs(_delta_phi_np(test_phi_np, ref_phi_np))

    bad = rel_pt > rel_tol

    _debug_log(
        obj,
        (
            f"{label}: mean(rel_pt)={rel_pt.mean():.6g}, max(rel_pt)={rel_pt.max():.6g}, "
            f"mean(|dphi|)={dphi.mean():.6g}, max(|dphi|)={dphi.max():.6g}, "
            f"n_bad(>{100.0 * rel_tol:.2f}%)={int(np.sum(bad))}/{len(rel_pt)}"
        ),
    )

    idx = np.where(bad)[0]
    if len(idx) == 0:
        idx = np.arange(min(max_entries, len(rel_pt)))
    else:
        idx = idx[:max_entries]

    for i in idx:
        _debug_log(
            obj,
            (
                f"{label}: event[{i}] "
                f"ref_pt={ref_pt_np[i]:.6g}, test_pt={test_pt_np[i]:.6g}, "
                f"rel_pt={rel_pt[i]:.6g}, ref_phi={ref_phi_np[i]:.6g}, "
                f"test_phi={test_phi_np[i]:.6g}, |dphi|={dphi[i]:.6g}"
            ),
        )


def get_evaluators(
    correction_set: correctionlib.highlevel.CorrectionSet,
    names: list[str],
    attrs: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """
    Helper function to get a list of correction evaluators from a
    correctionlib CorrectionSet object given a list of names.
    The names can refer to either simple or compound corrections.
    """
    available_keys = set(correction_set.keys()).union(set(correction_set.compound.keys()))
    corrected_names = []

    for name in names:
        if name in available_keys:
            corrected_names.append(name)
            continue

        closest_matches = difflib.get_close_matches(name, sorted(available_keys), n=1)
        if closest_matches:
            closest_match = closest_matches[0]
            logger.warning(
                "Correction '%s' not found. Using closest match '%s' instead.",
                name,
                closest_match,
            )
            corrected_names.append(closest_match)
        else:
            raise RuntimeError(
                f"Correction '{name}' not found and no close match available."
            )

    if attrs is not None and len(attrs) != len(corrected_names):
        raise ValueError(
            f"number of attribute dictionaries ({len(attrs)}) does not match "
            f"number of evaluator names ({len(corrected_names)})",
        )

    evaluators = []
    for i, name in enumerate(corrected_names):
        evaluator = (
            correction_set.compound[name]
            if name in correction_set.compound
            else correction_set[name]
        )

        if attrs is not None:
            for attr, value in attrs[i].items():
                setattr(evaluator, attr, value)

        evaluators.append(evaluator)

    return evaluators


def ak_evaluate(evaluator: correctionlib.highlevel.Correction, *args) -> float:
    """
    Evaluate a correctionlib Correction using one or more awkward arrays as inputs.
    """
    if not args:
        raise ValueError("Expected at least one argument.")

    ak_args = [arg for arg in args if isinstance(arg, ak.Array)]

    if ak_args:
        bc_args = ak.broadcast_arrays(*ak_args)
        flat_args = (
            np.asarray(ak.flatten(bc_arg, axis=None))
            for bc_arg in bc_args
        )
        output_layout_array = bc_args[0]
    else:
        flat_args = iter(())
        output_layout_array = None

    all_flat_args = [
        next(flat_args) if isinstance(arg, ak.Array) else arg
        for arg in args
    ]

    result = evaluator.evaluate(*all_flat_args)

    if output_layout_array is not None:
        result = layout_ak_array(result, output_layout_array)

    return result


#
# jet energy corrections
#

def get_jerc_file_default(self: Calibrator, external_files: DotDict) -> str:
    """
    Function to obtain external correction files for JEC and/or JER.
    """
    try_attrs = ("get_jec_config", "get_jer_config")
    jerc_config = None
    for try_attr in try_attrs:
        try:
            jerc_config = getattr(self, try_attr)()
        except AttributeError:
            continue
        else:
            break

    if jerc_config is None:
        raise ValueError(
            "could not retrieve jer/jec config, none of the following methods "
            f"were found: {try_attrs}",
        )

    ext_file_key = jerc_config.get("external_file_key", None)
    if ext_file_key is not None:
        return external_files[ext_file_key]

    if self.jet_name not in get_jerc_file_default.map_jet_name_file_key:
        available_keys = ", ".join(sorted(get_jerc_file_default.map_jet_name_file_key))
        raise ValueError(
            f"could not determine external file key for jet collection '{self.jet_name}', "
            f"name is not one of supported jet collections: {available_keys}",
        )

    ext_file_key = get_jerc_file_default.map_jet_name_file_key[self.jet_name]
    return external_files[ext_file_key]


get_jerc_file_default.map_jet_name_file_key = {
    "Jet": "jet_jerc",
}


def get_jec_config_default(self: Calibrator) -> DotDict:
    """
    Load config relevant to the jet energy corrections (JEC).
    """
    jec_cfg = self.config_inst.x.jec

    if self.jet_name not in jec_cfg:
        if self.jet_name == "Jet":
            logger.warning_once(
                f"{id(self)}_depr_jec_config",
                "config aux 'jec' does not contain key for input jet "
                f"collection '{self.jet_name}'. This may be due to "
                "an outdated config. Continuing under the assumption that "
                "the entire 'jec' entry refers to this jet collection. "
                "This assumption will be removed in future versions of "
                "columnflow, so please adapt the config according to the "
                "documentation to remove this warning and ensure future "
                "compatibility of the code.",
            )
            return jec_cfg

        raise ValueError(
            "config aux 'jec' does not contain key for input jet "
            f"collection '{self.jet_name}'.",
        )

    return jec_cfg[self.jet_name]


@calibrator(
    uses={
        "Jet.muonSubtrFactor",
        optional("Jet.muonSubtrDeltaPhi"),
        "Jet.chEmEF",
        "Jet.neEmEF",
        "run",
        optional("fixedGridRhoFastjetAll"),
        optional("Rho.fixedGridRhoFastjetAll"),
    },
    jet_name="Jet",
    met_name="PuppiMET",
    raw_met_name="RawPuppiMET",
    uncertainty_sources=None,
    propagate_met=False,
    get_jec_file=get_jerc_file_default,
    get_jec_config=get_jec_config_default,
    update_corrector_variables=(lambda self, corrector, variables: variables),
    debug=False,
    debug_max_events=5,
    debug_rel_tol=0.01,
)
def jec(
    self: Calibrator,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Perform nominal JEC and, when requested by the local task shift, exactly
    one JEC uncertainty source/direction.
    """
    jet_name = self.jet_name
    task = kwargs.get("task")
    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    _debug_log(
        self,
        (
            f"active JME shift in jec: kind={shift_kind}, "
            f"source={active_source}, direction={active_direction}"
        ),
    )

    def correct_jets(*, pt, eta, phi, area, rho, run, evaluator_key="jec"):
        variable_map = {
            "JetA": area,
            "JetEta": eta,
            "JetPt": pt,
            "JetPhi": phi,
            "Rho": ak.values_astype(rho, np.float32),
            "run": run,
        }

        full_correction = ak.ones_like(pt, dtype=np.float32)

        for corrector in self.evaluators[evaluator_key]:
            _variable_map = variable_map
            if callable(self.update_corrector_variables):
                _variable_map = variable_map.copy()
                _variable_map = self.update_corrector_variables(corrector, _variable_map)

            inputs = [_variable_map[inp.name] for inp in corrector.inputs]
            correction = ak_evaluate(corrector, *inputs)

            variable_map["JetPt"] = variable_map["JetPt"] * correction
            full_correction = full_correction * correction

            _debug_stats(
                self,
                f"{evaluator_key}:{getattr(corrector, 'level', 'unknown')} step",
                correction,
            )
            _debug_stats(
                self,
                f"{evaluator_key}:{getattr(corrector, 'level', 'unknown')} cumulative",
                full_correction,
            )

        return full_correction

    rho = (
        events.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in events.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    _debug_stats(self, f"{jet_name}.pt input", events[jet_name].pt)
    _debug_stats(self, "rho", rho)

    # Raw jet quantities.
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_raw",
        events[jet_name].pt * (1.0 - events[jet_name].rawFactor),
    )

    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_noMuRaw",
        events[jet_name].pt_raw * (1.0 - events[jet_name].muonSubtrFactor),
    )

    phi_noMuRaw = (
        events[jet_name].phi + events[jet_name].muonSubtrDeltaPhi
        if "muonSubtrDeltaPhi" in events[jet_name].fields
        else events[jet_name].phi
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.phi_noMuRaw",
        phi_noMuRaw,
    )

    # L1-only correction for Type-1 MET.
    jec_factors_l1 = correct_jets(
        pt=events[jet_name].pt_raw,
        eta=events[jet_name].eta,
        phi=events[jet_name].phi,
        area=events[jet_name].area,
        rho=rho,
        run=events.run,
        evaluator_key="jec_l1",
    )

    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_l1",
        events[jet_name].pt_raw * jec_factors_l1,
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_noMuL1",
        events[jet_name].pt_noMuRaw * jec_factors_l1,
    )

    # Full nominal correction.
    jec_factors = correct_jets(
        pt=events[jet_name].pt_raw,
        eta=events[jet_name].eta,
        phi=events[jet_name].phi,
        area=events[jet_name].area,
        rho=rho,
        run=events.run,
        evaluator_key="jec",
    )

    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt",
        events[jet_name].pt_raw * jec_factors,
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_noMuL1L2L3",
        events[jet_name].pt_noMuRaw * jec_factors,
    )

    raw_factor = ak.nan_to_num(
        1.0 - events[jet_name].pt_raw / events[jet_name].pt,
        nan=0.0,
    )
    events = set_ak_column_f32(events, f"{jet_name}.rawFactor", raw_factor)

    # Only evaluate the active JEC uncertainty.
    if shift_kind == "jec":
        if active_source not in self.evaluators["junc"]:
            raise RuntimeError(
                f"JEC evaluator for active source '{active_source}' was not loaded"
            )

        evaluator = self.evaluators["junc"][active_source]

        variable_map = {
            "JetEta": events[jet_name].eta,
            "JetPt": events[jet_name].pt,
        }
        inputs = [variable_map[inp.name] for inp in evaluator.inputs]
        jec_uncertainty = ak_evaluate(evaluator, *inputs)

        sign = 1.0 if active_direction == "up" else -1.0
        scale = 1.0 + sign * jec_uncertainty
        suffix = f"jec_{active_source}_{active_direction}"

        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_{suffix}",
            events[jet_name].pt * scale,
        )

        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_noMu_{suffix}",
            events[jet_name].pt_noMuL1L2L3 * scale,
        )

        _debug_stats(self, f"jec_uncertainty_{active_source}", jec_uncertainty)
        _debug_stats(self, f"{jet_name}.pt_{suffix}", events[jet_name][f"pt_{suffix}"])
        _debug_stats(
            self,
            f"{jet_name}.pt_noMu_{suffix}",
            events[jet_name][f"pt_noMu_{suffix}"],
        )

    return events


@jec.init
def jec_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()

    sources = self.uncertainty_sources
    if sources is None:
        sources = jec_cfg.uncertainty_sources or []
        self.uncertainty_sources = sources

    self.uses |= {
        "run",
        f"{self.jet_name}.pt",
        f"{self.jet_name}.eta",
        f"{self.jet_name}.phi",
        f"{self.jet_name}.area",
        f"{self.jet_name}.rawFactor",
        f"{self.jet_name}.muonSubtrFactor",
        optional(f"{self.jet_name}.muonSubtrDeltaPhi"),
        f"{self.jet_name}.chEmEF",
        f"{self.jet_name}.neEmEF",
    }

    # Keep the full declaration: it describes what the calibrator is capable of
    # producing across all shifts, not what one task invocation produces.
    self.produces |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.rawFactor",
        f"{self.jet_name}.pt_raw",
        f"{self.jet_name}.pt_l1",
        f"{self.jet_name}.pt_noMuRaw",
        f"{self.jet_name}.phi_noMuRaw",
        f"{self.jet_name}.pt_noMuL1",
        f"{self.jet_name}.pt_noMuL1L2L3",
    }

    # These branches are produced only in the matching local JEC-shift task.
    self.produces |= {
        optional(
            f"{self.jet_name}.pt_jec_{junc_name}_{junc_dir}"
        )
        for junc_name in sources
        for junc_dir in ("up", "down")
    }
    self.produces |= {
        optional(
            f"{self.jet_name}.pt_noMu_jec_{junc_name}_{junc_dir}"
        )
        for junc_name in sources
        for junc_dir in ("up", "down")
    }


@jec.requires
def jec_requires(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@jec.setup
def jec_setup(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    """
    Load JEC files and build nominal evaluators plus only the active JEC
    uncertainty evaluator.
    """
    jec_file = self.get_jec_file(reqs["external_files"].files)
    correction_set = load_correction_set(jec_file)

    jec_cfg = self.get_jec_config()

    def make_jme_keys(names, jec=jec_cfg, is_data=self.dataset_inst.is_data):
        if is_data and jec.get("data_per_era", True):
            if "data_per_era" not in jec:
                logger.warning_once(
                    f"{id(self)}_depr_jec_config_data_per_era",
                    "config aux 'jec' does not contain key 'data_per_era'. "
                    "Continuing under the assumption that JEC keys for data are era-specific.",
                )
            jec_era = self.dataset_inst.get_aux("jec_era", None)
            if jec_era is None:
                era = self.dataset_inst.get_aux("era", None)
                if era is None:
                    raise ValueError(
                        "JEC data key is requested to be era dependent, but neither jec_era nor era "
                        f"is set for dataset {self.dataset_inst.name}.",
                    )
                jec_era = "Run" + era

            jme_key = f"{jec.campaign}_{jec_era}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
        elif is_data:
            jme_key = f"{jec.campaign}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
        else:
            jme_key = f"{jec.campaign}_{jec.version}_MC_{{name}}_{jec.jet_type}"

        return [jme_key.format(name=name) for name in names]

    def get_main_levels() -> list[str]:
        key = "levels_DATA" if self.dataset_inst.is_data else "levels_MC"
        if key in jec_cfg:
            return list(jec_cfg[key])

        if "levels" in jec_cfg:
            logger.warning_once(
                f"{id(self)}_fallback_{key}",
                f"config aux 'jec' does not contain '{key}', falling back to 'levels'.",
            )
            return list(jec_cfg["levels"])

        raise ValueError(
            f"Could not find '{key}' in jec config for jet collection '{self.jet_name}'."
        )

    def get_l1_levels() -> list[str]:
        if "levels_for_type1_met" in jec_cfg:
            return list(jec_cfg["levels_for_type1_met"])

        l1_levels = [lvl for lvl in get_main_levels() if lvl.startswith("L1")]
        if l1_levels:
            return l1_levels

        raise ValueError(
            "Could not determine L1-only JEC levels. Please define "
            "'levels_for_type1_met' in the jec config."
        )

    jec_levels = get_main_levels()
    jec_l1_levels = get_l1_levels()

    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    active_sources = (
        [active_source]
        if shift_kind == "jec"
        else []
    )

    if active_source is not None and active_source not in self.uncertainty_sources:
        raise ValueError(
            f"active JEC source '{active_source}' is not configured in "
            f"uncertainty_sources={self.uncertainty_sources}"
        )

    jec_keys = make_jme_keys(jec_levels)
    jec_l1_keys = make_jme_keys(jec_l1_levels)
    junc_keys = make_jme_keys(active_sources, is_data=False)

    _debug_log(self, f"JEC file = {jec_file}")
    _debug_log(self, f"JEC main levels = {jec_levels}")
    _debug_log(self, f"JEC L1 levels = {jec_l1_levels}")
    _debug_log(
        self,
        (
            f"active JEC setup: kind={shift_kind}, source={active_source}, "
            f"direction={active_direction}"
        ),
    )
    _debug_log(self, f"JEC uncertainty keys loaded = {junc_keys}")

    self.evaluators = {
        "jec": get_evaluators(
            correction_set,
            jec_keys,
            attrs=[{"level": level} for level in jec_levels],
        ),
        "jec_l1": get_evaluators(
            correction_set,
            jec_l1_keys,
            attrs=[{"level": level} for level in jec_l1_levels],
        ),
        "junc": dict(
            zip(
                active_sources,
                get_evaluators(correction_set, junc_keys),
            )
        ),
    }


jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})
jec_ak4 = jec.derive("jec_ak4", cls_dict={"jet_name": "Jet"})
jec_ak4_nominal = jec_ak4.derive("jec_ak4_nominal", cls_dict={"uncertainty_sources": []})


def get_jer_config_default(self: Calibrator) -> DotDict:
    """
    Load config relevant to the jet energy resolution (JER) smearing.
    """
    jer_cfg = self.config_inst.x.jer

    if self.jet_name not in jer_cfg:
        if self.jet_name == "Jet":
            logger.warning_once(
                f"{id(self)}_depr_jer_config",
                "config aux 'jer' does not contain key for input jet "
                f"collection '{self.jet_name}'. This may be due to "
                "an outdated config. Continuing under the assumption that "
                "the entire 'jer' entry refers to this jet collection. "
                "This assumption will be removed in future versions of "
                "columnflow, so please adapt the config according to the "
                "documentation to remove this warning and ensure future "
                "compatibility of the code.",
            )
            return jer_cfg

        raise ValueError(
            "config aux 'jer' does not contain key for input jet "
            f"collection '{self.jet_name}'.",
        )

    return jer_cfg[self.jet_name]


#
# jet energy resolution smearing
#

@calibrator(
    uses={
        optional("Rho.fixedGridRhoFastjetAll"),
        optional("fixedGridRhoFastjetAll"),
    },
    jet_name="Jet",
    gen_jet_name="GenJet",
    met_name="PuppiMET",
    propagate_met=False,
    mc_only=True,
    deterministic_seed_index=-1,
    get_jer_file=get_jerc_file_default,
    get_jer_config=get_jer_config_default,
    get_jec_config=get_jec_config_default,
    jec_uncertainty_sources=None,
    gen_jet_matching_nominal=False,
    stochastic_smearing_mask=lambda self, jets: ak.ones_like(jets.pt, dtype=bool),
    debug=False,
    debug_max_events=5,
    debug_rel_tol=0.01,
)
def jer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Apply the same JER calculation as the old Type-1 implementation, but only
    for the branches required by the active CalibrateEvents shift.

    Numerical behavior is intentionally unchanged from the old code.
    """
    jet_name = self.jet_name
    gen_jet_name = self.gen_jet_name
    met_name = self.met_name

    if self.dataset_inst.is_data:
        raise ValueError("attempt to apply jet energy resolution smearing in data")

    task = kwargs.get("task")
    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    jer_nom, jer_up, jer_down = self.jer_variations

    # Old ordering, restricted to the branches needed for this task.
    active_variations = [jer_nom]
    active_postfixes = [""]

    if shift_kind == "jer":
        active_variations.append(active_direction)
        active_postfixes.append(f"_jer_{active_direction}")
    elif shift_kind == "jec":
        active_jec_var = f"jec_{active_source}_{active_direction}"
        active_variations.append(active_jec_var)
        active_postfixes.append(f"_{active_jec_var}")

    # Preserve nominal pre-JER pT.
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_unsmeared",
        events[jet_name].pt,
    )

    # JER up/down start from the nominal JEC pT, exactly as before.
    if shift_kind == "jer":
        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_jer_{active_direction}",
            events[jet_name].pt,
        )

        if self.propagate_met:
            events = set_ak_column_f32(
                events,
                f"{met_name}.pt_jer_{active_direction}",
                events[met_name].pt,
            )
            events = set_ak_column_f32(
                events,
                f"{met_name}.phi_jer_{active_direction}",
                events[met_name].phi,
            )

    # Same random-number generation as old.
    random_normal = (
        ak_random(
            0,
            1,
            events[jet_name].deterministic_seed,
            rand_func=self.deterministic_normal,
        )
        if self.deterministic_seed_index >= 0
        else ak_random(
            0,
            1,
            rand_func=np.random.Generator(
                np.random.SFC64(events.event.to_list())
            ).normal,
        )
    )

    rho = (
        events.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in events.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    variable_map = {
        "JetEta": events[jet_name].eta,
        "JetPt": events[jet_name].pt,
        "Rho": rho,
        "systematic": jer_nom,
    }

    # Nominal resolution.
    inputs = [
        variable_map[inp.name]
        for inp in self.evaluators["jer"].inputs
    ]
    jer_vals = {
        jer_nom: ak_evaluate(
            self.evaluators["jer"],
            *inputs,
        )
    }

    if shift_kind == "jer":
        jer_vals[active_direction] = jer_vals[jer_nom]

    if shift_kind == "jec":
        active_jec_var = f"jec_{active_source}_{active_direction}"
        _variable_map = variable_map | {
            "JetPt": events[jet_name][f"pt_{active_jec_var}"],
        }
        inputs = [
            _variable_map[inp.name]
            for inp in self.evaluators["jer"].inputs
        ]
        jer_vals[active_jec_var] = ak_evaluate(
            self.evaluators["jer"],
            *inputs,
        )

    # Scale factors.
    jersf = {}

    inputs = [
        variable_map[inp.name]
        for inp in self.evaluators["sf"].inputs
    ]
    jersf[jer_nom] = ak_evaluate(
        self.evaluators["sf"],
        *inputs,
    )

    if shift_kind == "jer":
        _variable_map = variable_map | {
            "systematic": active_direction,
        }
        inputs = [
            _variable_map[inp.name]
            for inp in self.evaluators["sf"].inputs
        ]
        jersf[active_direction] = ak_evaluate(
            self.evaluators["sf"],
            *inputs,
        )

    if shift_kind == "jec":
        active_jec_var = f"jec_{active_source}_{active_direction}"
        _variable_map = variable_map | {
            "JetPt": events[jet_name][f"pt_{active_jec_var}"],
        }
        inputs = [
            _variable_map[inp.name]
            for inp in self.evaluators["sf"].inputs
        ]
        jersf[active_jec_var] = ak_evaluate(
            self.evaluators["sf"],
            *inputs,
        )

    jer_arr = ak_concatenate_safe(
        [
            jer_vals[v][..., None]
            for v in active_variations
        ],
        axis=-1,
    )
    jersf_arr = ak_concatenate_safe(
        [
            jersf[v][..., None]
            for v in active_variations
        ],
        axis=-1,
    )

    # Stochastic smearing: identical to old.
    jersf2_m1 = jersf_arr**2 - 1
    add_smear = np.sqrt(
        ak.where(
            jersf2_m1 < 0,
            0,
            jersf2_m1,
        )
    )
    smear_factors_stochastic = ak.where(
        self.stochastic_smearing_mask(events[jet_name]),
        1.0 + random_normal * jer_arr * add_smear,
        1.0,
    )

    # Gen match: identical to old.
    gen_jet_idx = events[jet_name][self.gen_jet_idx_column]
    valid_gen_jet_idxs = ak.mask(
        gen_jet_idx,
        gen_jet_idx >= 0,
    )

    max_gen_jet_idx = ak.max(valid_gen_jet_idxs)
    padded_gen_jets = ak.pad_none(
        events[gen_jet_name],
        0 if max_gen_jet_idx is None else (max_gen_jet_idx + 1),
    )
    matched_gen_jet = padded_gen_jets[valid_gen_jet_idxs]

    if self.gen_jet_matching_nominal:
        match_pt = events[jet_name].pt
    else:
        pt_names = []
        for variation in active_variations:
            if variation in self.jer_variations:
                pt_names.append("pt")
            else:
                pt_names.append(f"pt_{variation}")

        match_pt = ak_concatenate_safe(
            [
                events[jet_name][pt_name][..., None]
                for pt_name in pt_names
            ],
            axis=-1,
        )

    pt_relative_diff = (
        1 - matched_gen_jet.pt / match_pt
    )

    is_matched_pt = (
        np.abs(pt_relative_diff)
        < 3 * jer_arr
    )
    is_matched_pt = ak.fill_none(
        is_matched_pt,
        False,
    )

    smear_factors_scaling = (
        1.0
        + (jersf_arr - 1.0)
        * pt_relative_diff
    )

    smear_factors = ak.where(
        is_matched_pt,
        smear_factors_scaling,
        smear_factors_stochastic,
    )

    # Keep the exact old behavior.
    smear_factors = ak.fill_none(
        smear_factors,
        0.0,
    )

    if self.propagate_met:
        jetsum_pt_before = {}
        jetsum_phi_before = {}

        for postfix in active_postfixes:
            jetsum_pt_before[postfix], jetsum_phi_before[postfix] = sum_transverse(
                events[jet_name][f"pt{postfix}"],
                events[jet_name].phi,
            )

    # Apply only active branches.
    for i, postfix in enumerate(active_postfixes):
        pt_name = f"pt{postfix}"

        events = set_ak_column_f32(
            events,
            f"{jet_name}.{pt_name}",
            events[jet_name][pt_name] * smear_factors[..., i],
        )

    if self.propagate_met:
        events = set_ak_column_f32(
            events,
            f"{met_name}.pt_unsmeared",
            events[met_name].pt,
        )
        events = set_ak_column_f32(
            events,
            f"{met_name}.phi_unsmeared",
            events[met_name].phi,
        )

        for postfix in active_postfixes:
            jetsum_pt_after, jetsum_phi_after = sum_transverse(
                events[jet_name][f"pt{postfix}"],
                events[jet_name].phi,
            )

            met_pt, met_phi = propagate_met(
                jetsum_pt_before[postfix],
                jetsum_phi_before[postfix],
                jetsum_pt_after,
                jetsum_phi_after,
                events[met_name][f"pt{postfix}"],
                events[met_name][f"phi{postfix}"],
            )

            events = set_ak_column_f32(
                events,
                f"{met_name}.pt{postfix}",
                met_pt,
            )
            events = set_ak_column_f32(
                events,
                f"{met_name}.phi{postfix}",
                met_phi,
            )

    return events


jer_ak4 = jer.derive(
    "jer_ak4",
    cls_dict={
        "jet_name": "Jet",
        "gen_jet_name": "GenJet",
    },
)


@jer.init
def jer_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()
    jec_sources = self.jec_uncertainty_sources
    if jec_sources is None:
        jec_sources = jec_cfg.uncertainty_sources or []
        self.jec_uncertainty_sources = jec_sources

    self.jec_variations = sum(
        (
            [f"jec_{unc}_up", f"jec_{unc}_down"]
            for unc in self.jec_uncertainty_sources
        ),
        [],
    )

    lower_first = lambda s: s[0].lower() + s[1:] if s else s
    self.gen_jet_idx_column = lower_first(self.gen_jet_name) + "Idx"

    # Restore the old JER bookkeeping expected by the numerical implementation.
    # The optimized runtime still evaluates only the active subset.
    self.jer_variations = ["nom", "up", "down"]
    self.postfixes = (
        ["", "_jer_up", "_jer_down"]
        + [f"_{jec_var}" for jec_var in self.jec_variations]
    )

    self.uses |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.eta",
        f"{self.jet_name}.phi",
        f"{self.jet_name}.{self.gen_jet_idx_column}",
        f"{self.gen_jet_name}.pt",
        f"{self.gen_jet_name}.eta",
        f"{self.gen_jet_name}.phi",
    }

    # The JEC-varied branches are only present in the corresponding local
    # calibration task, therefore mark them optional.
    self.uses |= {
        optional(f"{self.jet_name}.pt_{jec_var}")
        for jec_var in self.jec_variations
    }

    # Nominal outputs are always produced.
    self.produces |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.pt_unsmeared",
    }

    # Shifted branches are task-dependent capabilities.
    self.produces |= {
        optional(f"{self.jet_name}.pt_jer_{direction}")
        for direction in ("up", "down")
    }
    self.produces |= {
        optional(f"{self.jet_name}.pt_{jec_var}")
        for jec_var in self.jec_variations
    }

    if self.propagate_met:
        self.uses |= {
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
        }

        self.produces |= {
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
            f"{self.met_name}.pt_unsmeared",
            f"{self.met_name}.phi_unsmeared",
        }
        self.produces |= {
            optional(f"{self.met_name}.{var}_jer_{direction}")
            for var in ("pt", "phi")
            for direction in ("up", "down")
        }


@jer.requires
def jer_requires(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@jer.setup
def jer_setup(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    """
    Load JER files and build the resolution / scale-factor evaluators.
    """
    jer_file = self.get_jer_file(reqs["external_files"].files)
    correction_set = load_correction_set(jer_file)

    jer_cfg = self.get_jer_config()
    jer_keys = {
        "jer": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_PtResolution_{jer_cfg.jet_type}",
        "sf": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_ScaleFactor_{jer_cfg.jet_type}",
    }

    _debug_log(self, f"JER file = {jer_file}")
    _debug_log(self, f"JER keys = {jer_keys}")

    self.evaluators = {
        name: get_evaluators(correction_set, [key])[0]
        for name, key in jer_keys.items()
    }

    if self.deterministic_seed_index >= 0:
        idx = self.deterministic_seed_index
        bit_generator = np.random.SFC64

        def deterministic_normal(loc, scale, seed):
            return np.asarray([
                np.random.Generator(bit_generator(_seed)).normal(
                    _loc,
                    _scale,
                    size=idx + 1,
                )[-1]
                for _loc, _scale, _seed in zip(loc, scale, seed)
            ])

        self.deterministic_normal = deterministic_normal


#
# combined public calibrator:
# apply JEC + JER to jets internally, propagate to MET only here
#

@calibrator(
    uses={
        "run",
        optional("fixedGridRhoFastjetAll"),
        optional("Rho.fixedGridRhoFastjetAll"),
        optional("CorrT1METJet.rawPt"),
        optional("CorrT1METJet.eta"),
        optional("CorrT1METJet.area"),
        optional("CorrT1METJet.phi"),
        optional("CorrT1METJet.muonSubtrFactor"),
        optional("CorrT1METJet.muonSubtrDeltaPhi"),
        optional("CorrT1METJet.EmEF"),
    },
    jet_name="Jet",
    gen_jet_name="GenJet",
    met_name="PuppiMET",
    raw_met_name="RawPuppiMET",
    get_jec_file=get_jerc_file_default,
    get_jec_config=get_jec_config_default,
    get_jer_file=get_jerc_file_default,
    get_jer_config=get_jer_config_default,
    jec_uncertainty_sources=None,
    debug=False,
    debug_max_events=5,
    debug_rel_tol=0.01,
)
def jme(
    self: Calibrator,
    events: ak.Array,
    min_pt_met_prop: float = 15.0,
    **kwargs,
) -> ak.Array:
    """
    Apply JEC/JER internally and propagate only the active kinematic variation
    to PuppiMET.

    Nominal Type-1 MET is always built. In addition:
    - JEC shift task -> only that JEC MET variation is built.
    - JER shift task -> only that JER MET variation is built.
    """
    jet_name = self.jet_name
    met_name = self.met_name
    raw_met_name = self.raw_met_name
    task = kwargs.get("task")

    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    _debug_log(
        self,
        (
            f"starting jme: dataset={self.dataset_inst.name}, "
            f"is_data={self.dataset_inst.is_data}, is_mc={self.dataset_inst.is_mc}, "
            f"active_kind={shift_kind}, active_source={active_source}, "
            f"active_direction={active_direction}, min_pt_met_prop={min_pt_met_prop}"
        ),
    )

    input_met_pt = events[met_name].pt if met_name in events.fields else None
    input_met_phi = events[met_name].phi if met_name in events.fields else None

    # Internal JEC and JER are now shift-aware as well.
    events = self[self.jec_cls](events, **kwargs)

    if self.dataset_inst.is_mc:
        events = self[self.jer_cls](events, **kwargs)

    rho = (
        events.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in events.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    def has_corr_t1_collection(ev: ak.Array) -> bool:
        if "CorrT1METJet" not in ev.fields:
            return False
        needed = {
            "rawPt",
            "eta",
            "area",
            "phi",
            "muonSubtrFactor",
            "EmEF",
        }
        return needed.issubset(set(ev["CorrT1METJet"].fields))

    def evaluate_chain(evaluators, *, pt, eta, phi, area, rho, run):
        variable_map = {
            "JetA": area,
            "JetEta": eta,
            "JetPt": pt,
            "JetPhi": phi,
            "Rho": ak.values_astype(rho, np.float32),
            "run": run,
        }

        corr = ak.ones_like(pt, dtype=np.float32)
        for corrector in evaluators:
            inputs = [variable_map[inp.name] for inp in corrector.inputs]
            step = ak_evaluate(corrector, *inputs)
            variable_map["JetPt"] = variable_map["JetPt"] * step
            corr = corr * step

        return corr

    def jetsum(pt, phi, mask):
        return sum_transverse(pt[mask], phi[mask])

    corr_t1_present = has_corr_t1_collection(events)

    # ------------------------------------------------------------------
    # Nominal Type-1 MET
    # ------------------------------------------------------------------

    met_nom_pt = events[raw_met_name].pt
    met_nom_phi = events[raw_met_name].phi

    main_mask_nom = (
        (events[jet_name].pt_noMuL1L2L3 > min_pt_met_prop)
        & ((events[jet_name].chEmEF + events[jet_name].neEmEF) < 0.9)
    )

    main_l1_pt, main_l1_phi = jetsum(
        events[jet_name].pt_noMuL1,
        events[jet_name].phi_noMuRaw,
        main_mask_nom,
    )
    main_full_pt, main_full_phi = jetsum(
        events[jet_name].pt_noMuL1L2L3,
        events[jet_name].phi_noMuRaw,
        main_mask_nom,
    )

    met_nom_pt, met_nom_phi = propagate_met(
        main_l1_pt,
        main_l1_phi,
        main_full_pt,
        main_full_phi,
        met_nom_pt,
        met_nom_phi,
    )

    # CorrT1METJet nominal contribution.
    corr_nom_full_pt = None
    corr_nom_full_phi = None
    corr_pt_noMuL1L2L3_nom = None
    corr_phi_noMuRaw_nom = None

    if corr_t1_present:
        corr = events["CorrT1METJet"]

        corr_pt_noMuRaw = corr.rawPt * (1.0 - corr.muonSubtrFactor)
        corr_phi_noMuRaw = (
            corr.phi + corr.muonSubtrDeltaPhi
            if "muonSubtrDeltaPhi" in corr.fields
            else corr.phi
        )

        corr_l1_factor = evaluate_chain(
            self.corr_t1_evaluators["jec_l1"],
            pt=corr.rawPt,
            eta=corr.eta,
            phi=corr.phi,
            area=corr.area,
            rho=rho,
            run=events.run,
        )
        corr_full_factor = evaluate_chain(
            self.corr_t1_evaluators["jec"],
            pt=corr.rawPt,
            eta=corr.eta,
            phi=corr.phi,
            area=corr.area,
            rho=rho,
            run=events.run,
        )

        corr_pt_noMuL1 = corr_pt_noMuRaw * corr_l1_factor
        corr_pt_noMuL1L2L3 = corr_pt_noMuRaw * corr_full_factor

        corr_pt_noMuL1L2L3_nom = corr_pt_noMuL1L2L3
        corr_phi_noMuRaw_nom = corr_phi_noMuRaw

        corr_mask_nom = (
            (corr_pt_noMuL1L2L3 > min_pt_met_prop)
            & (corr.EmEF < 0.9)
        )

        corr_l1_pt, corr_l1_phi = jetsum(
            corr_pt_noMuL1,
            corr_phi_noMuRaw,
            corr_mask_nom,
        )
        corr_nom_full_pt, corr_nom_full_phi = jetsum(
            corr_pt_noMuL1L2L3,
            corr_phi_noMuRaw,
            corr_mask_nom,
        )

        met_nom_pt, met_nom_phi = propagate_met(
            corr_l1_pt,
            corr_l1_phi,
            corr_nom_full_pt,
            corr_nom_full_phi,
            met_nom_pt,
            met_nom_phi,
        )

    events = set_ak_column_f32(
        events,
        f"{met_name}.pt",
        met_nom_pt,
    )
    events = set_ak_column_f32(
        events,
        f"{met_name}.phi",
        met_nom_phi,
    )

    if input_met_pt is not None:
        _debug_compare_met(
            self,
            f"final propagated {met_name} vs input {met_name}",
            input_met_pt,
            input_met_phi,
            met_nom_pt,
            met_nom_phi,
        )

    # ------------------------------------------------------------------
    # Active JEC variation on MET
    # ------------------------------------------------------------------

    if shift_kind == "jec":
        unc = active_source
        direction = active_direction

        _debug_log(
            self,
            f"processing active JEC MET variation: {unc} {direction}",
        )

        met_var_pt = met_nom_pt
        met_var_phi = met_nom_phi

        # Main Jet collection.
        main_pt_var_name = f"pt_noMu_jec_{unc}_{direction}"
        main_pt_var = events[jet_name][main_pt_var_name]

        main_mask_var = (
            (main_pt_var > min_pt_met_prop)
            & ((events[jet_name].chEmEF + events[jet_name].neEmEF) < 0.9)
        )

        main_var_pt, main_var_phi = jetsum(
            main_pt_var,
            events[jet_name].phi_noMuRaw,
            main_mask_var,
        )

        met_var_pt, met_var_phi = propagate_met(
            main_full_pt,
            main_full_phi,
            main_var_pt,
            main_var_phi,
            met_var_pt,
            met_var_phi,
        )

        # CorrT1METJet collection.
        if corr_t1_present:
            corr = events["CorrT1METJet"]

            if unc not in self.corr_t1_evaluators["junc"]:
                raise RuntimeError(
                    f"CorrT1METJet evaluator for active source '{unc}' was not loaded"
                )

            unc_eval = self.corr_t1_evaluators["junc"][unc]

            unc_inputs = {
                "JetEta": corr.eta,
                "JetPt": corr_pt_noMuL1L2L3_nom,
            }
            corr_unc = ak_evaluate(
                unc_eval,
                *[unc_inputs[inp.name] for inp in unc_eval.inputs],
            )

            scale = (
                1.0 + corr_unc
                if direction == "up"
                else 1.0 - corr_unc
            )
            corr_pt_var = corr_pt_noMuL1L2L3_nom * scale

            corr_mask_var = (
                (corr_pt_var > min_pt_met_prop)
                & (corr.EmEF < 0.9)
            )

            corr_var_pt, corr_var_phi = jetsum(
                corr_pt_var,
                corr_phi_noMuRaw_nom,
                corr_mask_var,
            )

            met_var_pt, met_var_phi = propagate_met(
                corr_nom_full_pt,
                corr_nom_full_phi,
                corr_var_pt,
                corr_var_phi,
                met_var_pt,
                met_var_phi,
            )

        events = set_ak_column_f32(
            events,
            f"{met_name}.pt_jec_{unc}_{direction}",
            met_var_pt,
        )
        events = set_ak_column_f32(
            events,
            f"{met_name}.phi_jec_{unc}_{direction}",
            met_var_phi,
        )

    # ------------------------------------------------------------------
    # Active JER variation on MET
    # ------------------------------------------------------------------

    if self.dataset_inst.is_mc and shift_kind == "jer":
        direction = active_direction

        _debug_log(
            self,
            f"processing active JER MET variation: {direction}",
        )

        met_var_pt = met_nom_pt
        met_var_phi = met_nom_phi

        jer_scale = ak.nan_to_num(
            events[jet_name][f"pt_jer_{direction}"]
            / events[jet_name].pt,
            nan=1.0,
        )

        main_pt_jer = (
            events[jet_name].pt_noMuL1L2L3
            * jer_scale
        )

        main_mask_jer = (
            (main_pt_jer > min_pt_met_prop)
            & ((events[jet_name].chEmEF + events[jet_name].neEmEF) < 0.9)
        )

        main_jer_pt, main_jer_phi = jetsum(
            main_pt_jer,
            events[jet_name].phi_noMuRaw,
            main_mask_jer,
        )

        met_var_pt, met_var_phi = propagate_met(
            main_full_pt,
            main_full_phi,
            main_jer_pt,
            main_jer_phi,
            met_var_pt,
            met_var_phi,
        )

        events = set_ak_column_f32(
            events,
            f"{met_name}.pt_jer_{direction}",
            met_var_pt,
        )
        events = set_ak_column_f32(
            events,
            f"{met_name}.phi_jer_{direction}",
            met_var_phi,
        )

    return events


@jme.init
def jme_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()
    jec_sources = self.jec_uncertainty_sources
    if jec_sources is None:
        jec_sources = jec_cfg.uncertainty_sources or []
        self.jec_uncertainty_sources = jec_sources

    self.uses |= {
        f"{self.raw_met_name}.pt",
        f"{self.raw_met_name}.phi",
        "run",
        optional("fixedGridRhoFastjetAll"),
        optional("Rho.fixedGridRhoFastjetAll"),
        optional("CorrT1METJet.rawPt"),
        optional("CorrT1METJet.eta"),
        optional("CorrT1METJet.area"),
        optional("CorrT1METJet.phi"),
        optional("CorrT1METJet.muonSubtrFactor"),
        optional("CorrT1METJet.muonSubtrDeltaPhi"),
        optional("CorrT1METJet.EmEF"),
    }

    # Keep the full set of possible MET outputs declared for ColumnFlow.
    self.produces |= {
        f"{self.met_name}.pt",
        f"{self.met_name}.phi",
    }

    self.produces |= {
        optional(
            f"{self.met_name}.{shifted_var}_jec_{junc_name}_{junc_dir}"
        )
        for shifted_var in ("pt", "phi")
        for junc_name in jec_sources
        for junc_dir in ("up", "down")
    }

    if self.dataset_inst.is_mc:
        self.produces |= {
            optional(
                f"{self.met_name}.{shifted_var}_jer_{junc_dir}"
            )
            for shifted_var in ("pt", "phi")
            for junc_dir in ("up", "down")
        }

    def get_attrs(names, extra: dict[str, Any] | None = None):
        cls_dict = {}
        for name in names:
            value = getattr(self, name, UNSET)
            if value is not UNSET and value is not None:
                cls_dict[name] = value
        if extra:
            cls_dict.update(extra)
        return cls_dict

    self.jec_cls = jec.derive(
        f"{self.jet_name}_internal_jec_no_met",
        cls_dict=get_attrs(
            [
                "jet_name",
                "met_name",
                "raw_met_name",
                "get_jec_file",
                "get_jec_config",
                "debug",
                "debug_max_events",
                "debug_rel_tol",
            ],
            extra={
                "propagate_met": False,
                "uncertainty_sources": jec_sources,
            },
        ),
    )
    self.uses.add(self.jec_cls)

    if self.dataset_inst.is_mc:
        self.jer_cls = jer.derive(
            f"{self.jet_name}_internal_jer_no_met",
            cls_dict=get_attrs(
                [
                    "jet_name",
                    "gen_jet_name",
                    "met_name",
                    "get_jer_file",
                    "get_jer_config",
                    "get_jec_config",
                    "debug",
                    "debug_max_events",
                    "debug_rel_tol",
                ],
                extra={
                    "propagate_met": False,
                    "jec_uncertainty_sources": jec_sources,
                },
            ),
        )
        self.uses.add(self.jer_cls)


@jme.requires
def jme_requires(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@jme.setup
def jme_setup(
    self: Calibrator,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    """
    Build nominal CorrT1METJet JEC evaluators plus only the active JEC
    uncertainty evaluator.
    """
    jec_file = self.get_jec_file(reqs["external_files"].files)
    correction_set = load_correction_set(jec_file)

    jec_cfg = self.get_jec_config()

    def make_jme_keys(names, jec=jec_cfg, is_data=self.dataset_inst.is_data):
        if is_data and jec.get("data_per_era", True):
            jec_era = self.dataset_inst.get_aux("jec_era", None)
            if jec_era is None:
                era = self.dataset_inst.get_aux("era", None)
                if era is None:
                    raise ValueError(
                        "JEC data key is requested to be era dependent, but neither jec_era nor era "
                        f"is set for dataset {self.dataset_inst.name}.",
                    )
                jec_era = "Run" + era
            jme_key = f"{jec.campaign}_{jec_era}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
        elif is_data:
            jme_key = f"{jec.campaign}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
        else:
            jme_key = f"{jec.campaign}_{jec.version}_MC_{{name}}_{jec.jet_type}"

        return [jme_key.format(name=name) for name in names]

    def get_main_levels() -> list[str]:
        key = "levels_DATA" if self.dataset_inst.is_data else "levels_MC"
        if key in jec_cfg:
            return list(jec_cfg[key])
        if "levels" in jec_cfg:
            return list(jec_cfg["levels"])
        raise ValueError(f"Could not find '{key}' in jec config.")

    def get_l1_levels() -> list[str]:
        if "levels_for_type1_met" in jec_cfg:
            return list(jec_cfg["levels_for_type1_met"])
        l1_levels = [lvl for lvl in get_main_levels() if lvl.startswith("L1")]
        if l1_levels:
            return l1_levels
        raise ValueError(
            "Could not determine L1-only JEC levels. Please define "
            "'levels_for_type1_met' in the jec config."
        )

    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    active_sources = (
        [active_source]
        if shift_kind == "jec"
        else []
    )

    if active_source is not None and active_source not in self.jec_uncertainty_sources:
        raise ValueError(
            f"active JEC source '{active_source}' is not configured in "
            f"jec_uncertainty_sources={self.jec_uncertainty_sources}"
        )

    full_levels = get_main_levels()
    l1_levels = get_l1_levels()

    full_keys = make_jme_keys(full_levels)
    l1_keys = make_jme_keys(l1_levels)
    junc_keys = make_jme_keys(active_sources, is_data=False)

    _debug_log(self, f"JME CorrT1METJet JEC file = {jec_file}")
    _debug_log(
        self,
        (
            f"active CorrT1 JEC setup: kind={shift_kind}, source={active_source}, "
            f"direction={active_direction}"
        ),
    )
    _debug_log(self, f"JME CorrT1 uncertainty keys loaded = {junc_keys}")

    self.corr_t1_evaluators = {
        "jec": get_evaluators(
            correction_set,
            full_keys,
            attrs=[{"level": level} for level in full_levels],
        ),
        "jec_l1": get_evaluators(
            correction_set,
            l1_keys,
            attrs=[{"level": level} for level in l1_levels],
        ),
        "junc": dict(
            zip(
                active_sources,
                get_evaluators(correction_set, junc_keys),
            )
        ),
    }


jme_ak4 = jme.derive(
    "jme_ak4",
    cls_dict={
        "jet_name": "Jet",
        "gen_jet_name": "GenJet",
        "met_name": "PuppiMET",
        "raw_met_name": "RawPuppiMET",
    },
)

jme_ak4_debug = jme_ak4.derive(
    "jme_ak4_debug",
    cls_dict={
        "debug": True,
        "debug_max_events": 1,
        "debug_rel_tol": 0.01,
    },
)