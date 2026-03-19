# coding: utf-8

"""
Propagation of JEC and JER to PuppiMET.
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
    EMPTY_FLOAT,
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
    propagate_met=True,
    get_jec_file=get_jerc_file_default,
    get_jec_config=get_jec_config_default,
    update_corrector_variables=(lambda self, corrector, variables: variables),
)
def jec(
    self: Calibrator,
    events: ak.Array,
    min_pt_met_prop: float = 15.0,
    max_eta_met_prop: float = 5.2,
    **kwargs,
) -> ak.Array:
    """
    Perform jet energy corrections (JEC) and optionally propagate them to PuppiMET.
    """
    jet_name = self.jet_name
    met_name = self.met_name
    raw_met_name = self.raw_met_name

    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_raw",
        events[jet_name].pt * (1 - events[jet_name].rawFactor) *(1 - events[jet_name].muonSubtrFactor),
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

        return full_correction

    rho = (
        events.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in events.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    if self.propagate_met:
        jec_factors_subset_type1_met = correct_jets(
            pt=events[jet_name].pt_raw,
            eta=events[jet_name].eta,
            phi=events[jet_name].phi,
            area=events[jet_name].area,
            rho=rho,
            run=events.run,
            evaluator_key="jec_subset_type1_met",
        )

        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt",
            events[jet_name].pt_raw * jec_factors_subset_type1_met,
        )

        met_prop_mask = (
            (events[jet_name].pt > min_pt_met_prop) &
            (abs(events[jet_name].eta) < max_eta_met_prop) &
            ((events[jet_name].chEmEF + events[jet_name].neEmEF) < 0.9)
        )

        jetsum_pt_subset_type1_met, jetsum_phi_subset_type1_met = sum_transverse(
            events[jet_name][met_prop_mask].pt,
            events[jet_name][met_prop_mask].phi,
        )

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

    raw_factor = ak.nan_to_num(
        1 - events[jet_name].pt_raw / events[jet_name].pt,
        nan=0.0,
    )
    events = set_ak_column_f32(events, f"{jet_name}.rawFactor", raw_factor)

    if self.propagate_met:
        jetsum_pt_all_levels, jetsum_phi_all_levels = sum_transverse(
            events[jet_name][met_prop_mask].pt,
            events[jet_name][met_prop_mask].phi,
        )

        met_pt, met_phi = propagate_met(
            jetsum_pt_subset_type1_met,
            jetsum_phi_subset_type1_met,
            jetsum_pt_all_levels,
            jetsum_phi_all_levels,
            events[raw_met_name].pt,
            events[raw_met_name].phi,
        )

        events = set_ak_column_f32(events, f"{met_name}.pt", met_pt)
        events = set_ak_column_f32(events, f"{met_name}.phi", met_phi)

    variable_map = {
        "JetEta": events[jet_name].eta,
        "JetPt": events[jet_name].pt,
    }

    for name, evaluator in self.evaluators["junc"].items():
        inputs = [variable_map[inp.name] for inp in evaluator.inputs]
        jec_uncertainty = ak_evaluate(evaluator, *inputs)

        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_jec_{name}_up",
            events[jet_name].pt * (1.0 + jec_uncertainty),
        )
        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_jec_{name}_down",
            events[jet_name].pt * (1.0 - jec_uncertainty),
        )

        if self.propagate_met:
            jetsum_pt_up, jetsum_phi_up = sum_transverse(
                events[jet_name][met_prop_mask][f"pt_jec_{name}_up"],
                events[jet_name][met_prop_mask].phi,
            )
            jetsum_pt_down, jetsum_phi_down = sum_transverse(
                events[jet_name][met_prop_mask][f"pt_jec_{name}_down"],
                events[jet_name][met_prop_mask].phi,
            )

            met_pt_up, met_phi_up = propagate_met(
                jetsum_pt_all_levels,
                jetsum_phi_all_levels,
                jetsum_pt_up,
                jetsum_phi_up,
                met_pt,
                met_phi,
            )
            met_pt_down, met_phi_down = propagate_met(
                jetsum_pt_all_levels,
                jetsum_phi_all_levels,
                jetsum_pt_down,
                jetsum_phi_down,
                met_pt,
                met_phi,
            )

            events = set_ak_column_f32(events, f"{met_name}.pt_jec_{name}_up", met_pt_up)
            events = set_ak_column_f32(events, f"{met_name}.pt_jec_{name}_down", met_pt_down)
            events = set_ak_column_f32(events, f"{met_name}.phi_jec_{name}_up", met_phi_up)
            events = set_ak_column_f32(events, f"{met_name}.phi_jec_{name}_down", met_phi_down)

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
    }

    self.produces |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.rawFactor",
    }

    self.produces |= {
        f"{self.jet_name}.pt_jec_{junc_name}_{junc_dir}"
        for junc_name in sources
        for junc_dir in ("up", "down")
    }

    if self.propagate_met:
        self.uses |= {
            f"{self.raw_met_name}.pt",
            f"{self.raw_met_name}.phi",
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
        }
        self.produces |= {
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
        }
        self.produces |= {
            f"{self.met_name}.{shifted_var}_jec_{junc_name}_{junc_dir}"
            for shifted_var in ("pt", "phi")
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
    Load JEC files and build the corresponding evaluators.
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

    def get_type1_met_levels() -> list[str]:
        if "levels_for_type1_met" in jec_cfg:
            return list(jec_cfg["levels_for_type1_met"])

        logger.warning_once(
            f"{id(self)}_missing_levels_for_type1_met",
            "No 'levels_for_type1_met' found in jec config. "
            "Falling back to the full JEC levels.",
        )
        return jec_levels

    jec_levels = get_main_levels()

    if self.propagate_met:
        jec_subset_levels = get_type1_met_levels()
    else:
        jec_subset_levels = []

    jec_keys = make_jme_keys(jec_levels)
    junc_keys = make_jme_keys(self.uncertainty_sources, is_data=False)

    self.evaluators = {
        "jec": get_evaluators(
            correction_set,
            jec_keys,
            attrs=[{"level": level} for level in jec_levels],
        ),
        "junc": dict(
            zip(
                self.uncertainty_sources,
                get_evaluators(correction_set, junc_keys),
            )
        ),
    }

    if self.propagate_met:
        jec_subset_keys = make_jme_keys(jec_subset_levels)
        self.evaluators["jec_subset_type1_met"] = get_evaluators(
            correction_set,
            jec_subset_keys,
            attrs=[{"level": level} for level in jec_subset_levels],
        )


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
    propagate_met=True,
    mc_only=True,
    deterministic_seed_index=-1,
    get_jer_file=get_jerc_file_default,
    get_jer_config=get_jer_config_default,
    get_jec_config=get_jec_config_default,
    jec_uncertainty_sources=None,
    gen_jet_matching_nominal=False,
    stochastic_smearing_mask=lambda self, jets: ak.ones_like(jets.pt, dtype=bool),
)
def jer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Apply jet energy resolution smearing in MC and optionally propagate it to PuppiMET.
    """
    jet_name = self.jet_name
    gen_jet_name = self.gen_jet_name
    met_name = self.met_name

    if self.dataset_inst.is_data:
        raise ValueError("attempt to apply jet energy resolution smearing in data")

    jer_nom, jer_up, jer_down = self.jer_variations

    events = set_ak_column_f32(events, f"{jet_name}.pt_unsmeared", events[jet_name].pt)

    random_normal = (
        ak_random(0, 1, events[jet_name].deterministic_seed, rand_func=self.deterministic_normal)
        if self.deterministic_seed_index >= 0
        else ak_random(
            0,
            1,
            rand_func=np.random.Generator(np.random.SFC64(events.event.to_list())).normal,
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

    inputs = [variable_map[inp.name] for inp in self.evaluators["jer"].inputs]
    jer = {jer_nom: ak_evaluate(self.evaluators["jer"], *inputs)}

    jer[jer_up] = jer[jer_nom]
    jer[jer_down] = jer[jer_nom]

    for jec_var in self.jec_variations:
        _variable_map = variable_map | {"JetPt": events[jet_name][f"pt_{jec_var}"]}
        inputs = [_variable_map[inp.name] for inp in self.evaluators["jer"].inputs]
        jer[jec_var] = ak_evaluate(self.evaluators["jer"], *inputs)

    jersf = {}
    for jer_var in self.jer_variations:
        _variable_map = variable_map | {"systematic": jer_var}
        inputs = [_variable_map[inp.name] for inp in self.evaluators["sf"].inputs]
        jersf[jer_var] = ak_evaluate(self.evaluators["sf"], *inputs)

    for jec_var in self.jec_variations:
        _variable_map = variable_map | {"JetPt": events[jet_name][f"pt_{jec_var}"]}
        inputs = [_variable_map[inp.name] for inp in self.evaluators["sf"].inputs]
        jersf[jec_var] = ak_evaluate(self.evaluators["sf"], *inputs)

    jer = ak_concatenate_safe(
        [jer[v][..., None] for v in self.jer_variations + self.jec_variations],
        axis=-1,
    )
    jersf = ak_concatenate_safe(
        [jersf[v][..., None] for v in self.jer_variations + self.jec_variations],
        axis=-1,
    )

    jersf2_m1 = jersf**2 - 1
    add_smear = np.sqrt(ak.where(jersf2_m1 < 0, 0, jersf2_m1))

    smear_factors_stochastic = ak.where(
        self.stochastic_smearing_mask(events[jet_name]),
        1.0 + random_normal * jer * add_smear,
        1.0,
    )

    gen_jet_idx = events[jet_name][self.gen_jet_idx_column]
    valid_gen_jet_idxs = ak.mask(gen_jet_idx, gen_jet_idx >= 0)

    max_gen_jet_idx = ak.max(valid_gen_jet_idxs)
    padded_gen_jets = ak.pad_none(
        events[gen_jet_name],
        0 if max_gen_jet_idx is None else (max_gen_jet_idx + 1),
    )

    matched_gen_jet = padded_gen_jets[valid_gen_jet_idxs]

    if self.gen_jet_matching_nominal:
        match_pt = events[jet_name].pt
    else:
        pt_names = ["pt" for _ in self.jer_variations] + [f"pt_{jec_var}" for jec_var in self.jec_variations]
        match_pt = ak_concatenate_safe(
            [events[jet_name][pt_name][..., None] for pt_name in pt_names],
            axis=-1,
        )

    pt_relative_diff = 1 - matched_gen_jet.pt / match_pt

    is_matched_pt = np.abs(pt_relative_diff) < 3 * jer
    is_matched_pt = ak.fill_none(is_matched_pt, False)

    smear_factors_scaling = 1.0 + (jersf - 1.0) * pt_relative_diff

    smear_factors = ak.where(is_matched_pt, smear_factors_scaling, smear_factors_stochastic)
    smear_factors = ak.fill_none(smear_factors, 0.0)

    for direction in ["up", "down"]:
        events = set_ak_column_f32(events, f"{jet_name}.pt_jer_{direction}", events[jet_name].pt)
        if self.propagate_met:
            events = set_ak_column_f32(events, f"{met_name}.pt_jer_{direction}", events[met_name].pt)
            events = set_ak_column_f32(events, f"{met_name}.phi_jer_{direction}", events[met_name].phi)

    if self.propagate_met:
        jetsum_pt_before = {}
        jetsum_phi_before = {}
        for postfix in self.postfixes:
            jetsum_pt_before[postfix], jetsum_phi_before[postfix] = sum_transverse(
                events[jet_name][f"pt{postfix}"],
                events[jet_name].phi,
            )

    for i, postfix in enumerate(self.postfixes):
        pt_name = f"pt{postfix}"
        events = set_ak_column_f32(events, f"{jet_name}.{pt_name}", events[jet_name][pt_name] * smear_factors[..., i])

    if self.propagate_met:
        events = set_ak_column_f32(events, f"{met_name}.pt_unsmeared", events[met_name].pt)
        events = set_ak_column_f32(events, f"{met_name}.phi_unsmeared", events[met_name].phi)

        for postfix in self.postfixes:
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
            events = set_ak_column_f32(events, f"{met_name}.pt{postfix}", met_pt)
            events = set_ak_column_f32(events, f"{met_name}.phi{postfix}", met_phi)

    return events


jer_ak4 = jer.derive("jer_ak4", cls_dict={"jet_name": "Jet", "gen_jet_name": "GenJet"})


@jer.init
def jer_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()
    jec_sources = self.jec_uncertainty_sources
    if jec_sources is None:
        jec_sources = jec_cfg.uncertainty_sources or []
        self.jec_uncertainty_sources = jec_sources

    self.jec_variations = sum(
        ([f"jec_{unc}_up", f"jec_{unc}_down"] for unc in self.jec_uncertainty_sources),
        [],
    )

    jet_jec_columns = {
        f"{self.jet_name}.pt_{jec_source}"
        for jec_source in self.jec_variations
    }
    met_jec_columns = {
        f"{self.met_name}.{var}_{jec_source}"
        for var in ("pt", "phi")
        for jec_source in self.jec_variations
    }

    lower_first = lambda s: s[0].lower() + s[1:] if s else s
    self.gen_jet_idx_column = lower_first(self.gen_jet_name) + "Idx"

    self.jer_variations = ["nom", "up", "down"]
    self.postfixes = ["", "_jer_up", "_jer_down"] + [f"_{jec_var}" for jec_var in self.jec_variations]

    self.uses |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.eta",
        f"{self.jet_name}.phi",
        f"{self.jet_name}.{self.gen_jet_idx_column}",
        f"{self.gen_jet_name}.pt",
        f"{self.gen_jet_name}.eta",
        f"{self.gen_jet_name}.phi",
    }
    if jec_sources:
        self.uses |= jet_jec_columns

    self.produces |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.pt_unsmeared",
        f"{self.jet_name}.pt_jer_up",
        f"{self.jet_name}.pt_jer_down",
    }
    if jec_sources:
        self.produces |= jet_jec_columns

    if self.propagate_met:
        self.uses |= {
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
        }
        
        if jec_sources:
            self.uses |= met_jec_columns

        self.produces |= {
            f"{self.met_name}.pt",
            f"{self.met_name}.phi",
            f"{self.met_name}.pt_jer_up",
            f"{self.met_name}.phi_jer_up",
            f"{self.met_name}.pt_jer_down",
            f"{self.met_name}.phi_jer_down",
            f"{self.met_name}.pt_unsmeared",
            f"{self.met_name}.phi_unsmeared",
        }
        if jec_sources:
            self.produces |= met_jec_columns


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
    Load JER files and build the corresponding evaluators.
    """
    jer_file = self.get_jer_file(reqs["external_files"].files)
    correction_set = load_correction_set(jer_file)

    jer_cfg = self.get_jer_config()
    jer_keys = {
        "jer": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_PtResolution_{jer_cfg.jet_type}",
        "sf": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_ScaleFactor_{jer_cfg.jet_type}",
    }

    self.evaluators = {
        name: get_evaluators(correction_set, [key])[0]
        for name, key in jer_keys.items()
    }

    if self.deterministic_seed_index >= 0:
        idx = self.deterministic_seed_index
        bit_generator = np.random.SFC64

        def deterministic_normal(loc, scale, seed):
            return np.asarray([
                np.random.Generator(bit_generator(_seed)).normal(_loc, _scale, size=idx + 1)[-1]
                for _loc, _scale, _seed in zip(loc, scale, seed)
            ])

        self.deterministic_normal = deterministic_normal


#
# combined calibrator: compute JEC+JER internally, propagate to MET once,
# do not modify the output Jet collection
#

@calibrator(
    jet_name="Jet",
    gen_jet_name="GenJet",
    met_name="PuppiMET",
    raw_met_name="RawPuppiMET",
    propagate_met=True,
    get_jec_file=get_jerc_file_default,
    get_jec_config=get_jec_config_default,
    get_jer_file=get_jerc_file_default,
    get_jer_config=get_jer_config_default,
    jec_uncertainty_sources=None,
)
def jets_puppimet_only(
    self: Calibrator,
    events: ak.Array,
    min_pt_met_prop: float = 15.0,
    max_eta_met_prop: float = 5.2,
    **kwargs,
) -> ak.Array:
    """
    Compute JEC and JER on a temporary jet collection and propagate the final
    jet changes to PuppiMET exactly once, without modifying the output Jet
    collection in *events*.
    """
    jet_name = self.jet_name
    met_name = self.met_name

    def get_jetsum(ev: ak.Array, pt_field: str = "pt") -> tuple[ak.Array, ak.Array]:
        jets = ev[jet_name]
        pt = jets[pt_field]
        phi = jets.phi
        eta = jets.eta

        mask = (pt > min_pt_met_prop) & (abs(eta) < max_eta_met_prop) & ((jets.chEmEF + jets.neEmEF) < 0.9)

        return sum_transverse(
            pt[mask],
            phi[mask],
        )

    tmp_events = events

    tmp_events = self[self.jec_cls](tmp_events, **kwargs)

    if self.dataset_inst.is_mc:
        tmp_events = self[self.jer_cls](tmp_events, **kwargs)

    jetsum_pt_before, jetsum_phi_before = get_jetsum(events, "pt")
    jetsum_pt_after, jetsum_phi_after = get_jetsum(tmp_events, "pt")

    met_input_pt = events[met_name].pt
    met_input_phi = events[met_name].phi

    met_nominal_pt, met_nominal_phi = propagate_met(
        jetsum_pt_before,
        jetsum_phi_before,
        jetsum_pt_after,
        jetsum_phi_after,
        met_input_pt,
        met_input_phi,
    )

    events = set_ak_column_f32(events, f"{met_name}.pt", met_nominal_pt)
    events = set_ak_column_f32(events, f"{met_name}.phi", met_nominal_phi)

    for unc in self.jec_uncertainty_sources:
        for direction in ("up", "down"):
            pt_field = f"pt_jec_{unc}_{direction}"
            jetsum_pt_var, jetsum_phi_var = get_jetsum(tmp_events, pt_field)

            met_pt_var, met_phi_var = propagate_met(
                jetsum_pt_after,
                jetsum_phi_after,
                jetsum_pt_var,
                jetsum_phi_var,
                met_nominal_pt,
                met_nominal_phi,
            )

            events = set_ak_column_f32(events, f"{met_name}.pt_jec_{unc}_{direction}", met_pt_var)
            events = set_ak_column_f32(events, f"{met_name}.phi_jec_{unc}_{direction}", met_phi_var)

    if self.dataset_inst.is_mc:
        for direction in ("up", "down"):
            pt_field = f"pt_jer_{direction}"
            jetsum_pt_var, jetsum_phi_var = get_jetsum(tmp_events, pt_field)

            met_pt_var, met_phi_var = propagate_met(
                jetsum_pt_after,
                jetsum_phi_after,
                jetsum_pt_var,
                jetsum_phi_var,
                met_nominal_pt,
                met_nominal_phi,
            )

            events = set_ak_column_f32(events, f"{met_name}.pt_jer_{direction}", met_pt_var)
            events = set_ak_column_f32(events, f"{met_name}.phi_jer_{direction}", met_phi_var)

    return events


@jets_puppimet_only.init
def jets_puppimet_only_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()
    jec_sources = self.jec_uncertainty_sources
    if jec_sources is None:
        jec_sources = jec_cfg.uncertainty_sources or []
        self.jec_uncertainty_sources = jec_sources

    self.uses |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.eta",
        f"{self.jet_name}.phi",
        f"{self.jet_name}.area",
        f"{self.jet_name}.rawFactor",
        f"{self.met_name}.pt",
        f"{self.met_name}.phi",
    }

    if self.dataset_inst.is_mc:
        lower_first = lambda s: s[0].lower() + s[1:] if s else s
        gen_jet_idx_column = lower_first(self.gen_jet_name) + "Idx"
        self.uses.add(f"{self.jet_name}.{gen_jet_idx_column}")
        self.uses |= {
            f"{self.gen_jet_name}.pt",
            f"{self.gen_jet_name}.eta",
            f"{self.gen_jet_name}.phi",
        }

    self.produces |= {
        f"{self.met_name}.pt",
        f"{self.met_name}.phi",
    }

    self.produces |= {
        f"{self.met_name}.{shifted_var}_jec_{junc_name}_{junc_dir}"
        for shifted_var in ("pt", "phi")
        for junc_name in jec_sources
        for junc_dir in ("up", "down")
    }

    if self.dataset_inst.is_mc:
        self.produces |= {
            f"{self.met_name}.{shifted_var}_jer_{junc_dir}"
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
        f"{self.jet_name}_internal_no_met_jec",
        cls_dict=get_attrs(
            ["jet_name", "met_name", "raw_met_name", "get_jec_file", "get_jec_config"],
            extra={"propagate_met": False},
        ),
    )
    self.uses.add(self.jec_cls)

    if self.dataset_inst.is_mc:
        self.jer_cls = jer.derive(
            f"{self.jet_name}_internal_no_met_jer",
            cls_dict=get_attrs(
                [
                    "jet_name",
                    "gen_jet_name",
                    "met_name",
                    "get_jer_file",
                    "get_jer_config",
                    "get_jec_config",
                    "jec_uncertainty_sources",
                ],
                extra={"propagate_met": False},
            ),
        )
        self.uses.add(self.jer_cls)


jets_puppimet_only_ak4 = jets_puppimet_only.derive(
    "jets_puppimet_only_ak4",
    cls_dict={
        "jet_name": "Jet",
        "gen_jet_name": "GenJet",
        "met_name": "PuppiMET",
        "raw_met_name": "RawPuppiMET",
        "propagate_met": True,
    },
)