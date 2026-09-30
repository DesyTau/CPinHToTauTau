# coding: utf-8

"""
Jet energy corrections and jet resolution smearing.
"""

from __future__ import annotations

import difflib
import functools

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.util import ak_random
from columnflow.production.util import attach_coffea_behavior
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


def get_active_jme_shift(task: law.Task | None) -> tuple[str, str | None, str | None]:
    """
    Resolve the JEC/JER variation that belongs to the current calibration task.

    Returns a tuple ``(kind, source, direction)`` where:

    - nominal or unrelated shift -> ``("nominal", None, None)``
    - JEC source shift -> ``("jec", "<source>", "up"|"down")``
    - JER shift -> ``("jer", None, "up"|"down")``

    JEC/JER are local shifts of ``CalibrateEvents`` in the split-shift setup.
    Falling back to ``global_shift_inst`` keeps this helper usable in direct
    tests and in tasks that do not expose ``local_shift_inst``.
    """
    if task is None:
        return "nominal", None, None

    shift_inst = getattr(task, "local_shift_inst", None)
    if shift_inst is None:
        shift_inst = getattr(task, "global_shift_inst", None)

    if shift_inst is None or shift_inst.name == "nominal":
        return "nominal", None, None

    name = shift_inst.name

    if shift_inst.has_tag("jec"):
        source = shift_inst.x("jec_source", None)
        if source is None:
            if not name.startswith("jec_"):
                raise ValueError(f"cannot determine JEC source from shift '{name}'")
            source = name[len("jec_"):]

            if source.endswith("_up"):
                source = source[:-len("_up")]
            elif source.endswith("_down"):
                source = source[:-len("_down")]

        if name.endswith("_up"):
            direction = "up"
        elif name.endswith("_down"):
            direction = "down"
        else:
            raise ValueError(f"cannot determine JEC direction from shift '{name}'")

        return "jec", source, direction

    if shift_inst.has_tag("jer"):
        if name.endswith("_up"):
            direction = "up"
        elif name.endswith("_down"):
            direction = "down"
        else:
            raise ValueError(f"cannot determine JER direction from shift '{name}'")

        return "jer", None, direction

    # Other calibration shifts use nominal JEC/JER.
    return "nominal", None, None

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
    Evaluate a :external+correctionlib:py:class:`correctionlib.highlevel.Correction`
    using one or more :external+ak:py:class:`awkward arrays <ak.Array>` as inputs.

    :param evaluator: Evaluator instance
    :raises ValueError: If no :external+ak:py:class:`awkward arrays <ak.Array>` are provided
    :return: The correction factor derived from the input arrays
    """
    if not args:
        raise ValueError("Expected at least one argument.")

    ak_args = [
        arg for arg in args if isinstance(arg, ak.Array)
    ]

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

# define default functions for jec calibrator
def get_jerc_file_default(self: Calibrator, external_files: DotDict) -> str:
    """
    Function to obtain external correction files for JEC and/or JER.

    By default, this function extracts the location of the jec correction
    files from the current config instance *config_inst*. The key of the
    external file depends on the jet collection. For ``Jet`` (AK4 jets), this
    resolves to ``jet_jerc``, and for ``FatJet`` it is resolved to
    ``fat_jet_jerc``.

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "jet_jerc": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/JME/2017_UL/jet_jerc.json.gz",
            "fat_jet_jerc": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/JME/2017_UL/fatJet_jerc.json.gz",
        })

    :param external_files: Dictionary containing the information about the file location
    :return: path or url to correction file(s)
    """  # noqa

    # get config
    try_attrs = ("get_jec_config", "get_jer_config")
    jerc_config = None
    for try_attr in try_attrs:
        try:
            jerc_config = getattr(self, try_attr)()
        except AttributeError:
            continue
        else:
            break

    # fail if not found
    if jerc_config is None:
        raise ValueError(
            "could not retrieve jer/jec config, none of the following methods "
            f"were found: {try_attrs}",
        )

    # first check config for user-supplied `external_file_key`
    ext_file_key = jerc_config.get("external_file_key", None)
    if ext_file_key is not None:
        return external_files[ext_file_key]

    # if not found, try to resolve from jet collection name and fail if not standard NanoAOD
    if self.jet_name not in get_jerc_file_default.map_jet_name_file_key:
        available_keys = ", ".join(sorted(get_jerc_file_default.map_jet_name_file_key))
        raise ValueError(
            f"could not determine external file key for jet collection '{self.jet_name}', "
            f"name is not one of standard NanoAOD jet collections: {available_keys}",
        )

    # return external file
    ext_file_key = get_jerc_file_default.map_jet_name_file_key[self.jet_name]
    return external_files[ext_file_key]


# default external file keys for known jet collections
get_jerc_file_default.map_jet_name_file_key = {
    "Jet": "jet_jerc",
}


def get_jec_config_default(self: Calibrator) -> DotDict:
    """
    Load config relevant to the jet energy corrections (JEC).

    By default, this is extracted from the current *config_inst*,
    assuming the JEC configurations are stored under the 'jec'
    aux key. Separate configurations should be specified for each
    jet collection, using the collection name as a key. For example,
    the configuration for the default jet collection ``Jet`` will
    be retrieved from the following config entry:

    .. code-block:: python

        self.config_inst.x.jec.Jet

    Used in :py:meth:`~.jec.setup_func`.

    :return: Dictionary containing configuration for jet energy calibration
    """
    jec_cfg = self.config_inst.x.jec

    # check for old-style config
    if self.jet_name not in jec_cfg:
        # if jet collection is `Jet`, issue deprecation warning
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

        # otherwise raise exception
        raise ValueError(
            "config aux 'jec' does not contain key for input jet "
            f"collection '{self.jet_name}'.",
        )

    return jec_cfg[self.jet_name]


@calibrator(
    uses={
        "run",
        optional("fixedGridRhoFastjetAll"),
        optional("Rho.fixedGridRhoFastjetAll"),
        attach_coffea_behavior,
    },
    # name of the jet collection to calibrate
    jet_name="Jet",
    # custom uncertainty sources, defaults to config when empty
    uncertainty_sources=None,
    # function to determine the correction file
    get_jec_file=get_jerc_file_default,
    # function to determine the jec configuration dict
    get_jec_config=get_jec_config_default,
    # function to update variables before jec corrector call
    update_corrector_variables=(lambda self, corrector, variables: variables),
)
def jec(
    self: Calibrator,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Apply nominal JEC and, for a JEC-shifted CalibrateEvents task, calculate
    only the requested uncertainty source and direction.

    Example:
        local_shift = jec_Regrouped_BBEC1_up

    computes:
        - nominal JEC
        - Regrouped_BBEC1 uncertainty
        - Jet.pt_jec_Regrouped_BBEC1_up
        - Jet.mass_jec_Regrouped_BBEC1_up

    No other JEC uncertainty source or direction is evaluated.
    """
    jet_name = self.jet_name
    task = kwargs.get("task")
    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    # calculate uncorrected pt and mass
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_raw",
        events[jet_name].pt * (1 - events[jet_name].rawFactor),
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.mass_raw",
        events[jet_name].mass * (1 - events[jet_name].rawFactor),
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
                _variable_map = self.update_corrector_variables(
                    corrector,
                    _variable_map,
                )

            inputs = [
                _variable_map[inp.name]
                for inp in corrector.inputs
            ]
            correction = ak_evaluate(
                corrector,
                *inputs,
            )

            # Subsequent JEC levels must see the already corrected pT.
            variable_map["JetPt"] = (
                variable_map["JetPt"] * correction
            )
            full_correction = (
                full_correction * correction
            )

        return full_correction

    rho = (
        events.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in events.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    # nominal/full JEC
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
        f"{jet_name}.mass",
        events[jet_name].mass_raw * jec_factors,
    )

    raw_factor = ak.nan_to_num(
        1 - events[jet_name].pt_raw / events[jet_name].pt,
        nan=0.0,
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.rawFactor",
        raw_factor,
    )

    events = self[attach_coffea_behavior](
        events,
        collections=[jet_name],
        **kwargs,
    )

    # Nothing else is needed for nominal or JER-shifted tasks.
    if shift_kind != "jec":
        return events

    if active_source not in self.evaluators["junc"]:
        raise RuntimeError(
            f"JEC evaluator for active source '{active_source}' was not loaded. "
            f"Available sources: {sorted(self.evaluators['junc'])}"
        )

    # Keep the same uncertainty-evaluation convention as the previous
    # implementation to make this change purely an execution optimization.
    variable_map = {
        "JetEta": events[jet_name].eta,
        "JetPt": events[jet_name].pt_raw,
    }

    evaluator = self.evaluators["junc"][active_source]
    inputs = [
        variable_map[inp.name]
        for inp in evaluator.inputs
    ]
    uncertainty = ak_evaluate(
        evaluator,
        *inputs,
    )

    sign = 1.0 if active_direction == "up" else -1.0
    scale = 1.0 + sign * uncertainty
    suffix = f"_jec_{active_source}_{active_direction}"

    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt{suffix}",
        events[jet_name].pt * scale,
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.mass{suffix}",
        events[jet_name].mass * scale,
    )

    return events


@jec.init
def jec_init(self: Calibrator, **kwargs) -> None:
    jec_cfg = self.get_jec_config()

    sources = self.uncertainty_sources
    if sources is None:
        sources = jec_cfg.uncertainty_sources or []
        self.uncertainty_sources = sources

    self.uses.add("run")
    self.uses.add(f"{self.jet_name}.{{pt,eta,phi,mass,area,rawFactor}}")

    self.produces.add(f"{self.jet_name}.{{pt,mass,rawFactor}}")

    # Shifted JEC branches are produced only by the corresponding local-shift
    # task. Keep them declared, but optional, so ColumnFlow does not require
    # every source/direction after each call.
    self.produces |= {
        optional(
            f"{self.jet_name}.{shifted_var}_jec_{junc_name}_{junc_dir}"
        )
        for shifted_var in ("pt", "mass")
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
    Load nominal JEC evaluators and only the uncertainty evaluator required by
    the active local JEC shift.
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
                    "This may be due to an outdated config. Continuing under the assumption that "
                    "JEC keys for data are era-specific. "
                    "This assumption will be removed in future versions of "
                    "columnflow, so please adapt the config according to the "
                    "documentation to remove this warning and ensure future compatibility.",
                )

            jec_era = self.dataset_inst.get_aux("jec_era", None)

            if jec_era is None:
                era = self.dataset_inst.get_aux("era", None)

                if era is None:
                    raise ValueError(
                        "JEC data key is requested to be era dependent, but neither jec_era nor era "
                        f"auxiliary is set for dataset {self.dataset_inst.name}.",
                    )

                jec_era = "Run" + era

            jme_key = (
                f"{jec.campaign}_{jec_era}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
            )
        elif is_data:
            jme_key = (
                f"{jec.campaign}_{jec.version}_DATA_{{name}}_{jec.jet_type}"
            )
        else:
            jme_key = (
                f"{jec.campaign}_{jec.version}_MC_{{name}}_{jec.jet_type}"
            )

        return [
            jme_key.format(name=name)
            for name in names
        ]

    if self.dataset_inst.is_data:
        levels = list(jec_cfg.levels_DATA)
    else:
        levels = list(jec_cfg.levels_MC)

    jec_keys = make_jme_keys(levels)

    shift_kind, active_source, _ = get_active_jme_shift(task)

    active_sources = []
    if shift_kind == "jec":
        if active_source not in self.uncertainty_sources:
            raise ValueError(
                f"active JEC source '{active_source}' is not configured for '{self.jet_name}'. "
                f"Configured sources: {self.uncertainty_sources}"
            )

        active_sources = [
            active_source,
        ]

    # JEC uncertainties are stored with MC-style keys also when the nominal
    # JEC chain is configured for data.
    junc_keys = make_jme_keys(
        active_sources,
        is_data=False,
    )

    self.evaluators = {
        "jec": get_evaluators(
            correction_set,
            jec_keys,
            attrs=[
                {"level": level}
                for level in levels
            ],
        ),
        "junc": dict(
            zip(
                active_sources,
                get_evaluators(
                    correction_set,
                    junc_keys,
                ),
            )
        ),
    }


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})

# explicit calibrators for standard jet collections
jec_ak4 = jec.derive("jec_ak4", cls_dict={"jet_name": "Jet"})
jec_ak4_nominal = jec_ak4.derive("jec_ak4", cls_dict={"uncertainty_sources": []})



def get_jer_config_default(self: Calibrator) -> DotDict:
    """
    Load config relevant to the jet energy resolution (JER) smearing.

    By default, this is extracted from the current *config_inst*,
    assuming the JER configurations are stored under the 'jer'
    aux key. Separate configurations should be specified for each
    jet collection, using the collection name as a key. For example,
    the configuration for the default jet collection ``Jet`` will
    be retrieved from the following config entry:

    .. code-block:: python

        self.config_inst.x.jer.Jet

    Used in :py:meth:`~.jer.setup_func`.

    :return: Dictionary containing configuration for JER smearing
    """
    jer_cfg = self.config_inst.x.jer

    # check for old-style config
    if self.jet_name not in jer_cfg:
        # if jet collection is `Jet`, issue deprecation warning
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

        # otherwise raise exception
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
        attach_coffea_behavior,
    },
    # name of the jet collection to smear
    jet_name="Jet",
    # name of the associated gen jet collection
    gen_jet_name="GenJet",
    # only run on mc
    mc_only=True,
    # use deterministic seeds for random smearing and
    # take the "index"-th random number per seed when not -1
    deterministic_seed_index=-1,
    # function to determine the correction file
    get_jer_file=get_jerc_file_default,
    # function to determine the jer configuration dict
    get_jer_config=get_jer_config_default,
    # function to determine the jec configuration dict
    get_jec_config=get_jec_config_default,
    # jec uncertainty sources to propagate jer to, defaults to config when empty
    jec_uncertainty_sources=None,
    # whether gen jet matching should be performed relative to nominal jet pt
    gen_jet_matching_nominal=False,
    # regions where stochastic smearing is applied
    stochastic_smearing_mask=lambda self, jets: ak.ones_like(jets.pt, dtype=bool),
)
def jer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Apply the same JER calculation as the old implementation, but evaluate only
    the branches required by the active CalibrateEvents shift.

    The numerical recipe is intentionally kept identical to the old code:
    - same JER resolution and SF evaluators,
    - same random-number generation,
    - same gen-jet matching criterion,
    - same hybrid scaling/stochastic formulas,
    - same ak.fill_none(..., 0.0) behavior.

    Runtime branches:
    - nominal: nominal only;
    - JEC shift: nominal + the requested JEC branch;
    - JER shift: nominal + the requested JER branch.
    """
    jet_name = self.jet_name
    gen_jet_name = self.gen_jet_name

    if self.dataset_inst.is_data:
        raise ValueError("attempt to apply jet energy resolution smearing in data")

    task = kwargs.get("task")
    shift_kind, active_source, active_direction = get_active_jme_shift(task)

    jer_nom, jer_up, jer_down = self.jer_variations

    # Keep the old branch ordering, but retain only the branches needed by this task.
    active_variations = [jer_nom]
    active_postfixes = [""]

    if shift_kind == "jer":
        active_variations.append(active_direction)
        active_postfixes.append(f"_jer_{active_direction}")
    elif shift_kind == "jec":
        active_jec_var = f"jec_{active_source}_{active_direction}"
        active_variations.append(active_jec_var)
        active_postfixes.append(f"_{active_jec_var}")

    # Save the nominal JEC values before any smearing, exactly as in the old code.
    events = set_ak_column_f32(
        events,
        f"{jet_name}.pt_unsmeared",
        events[jet_name].pt,
    )
    events = set_ak_column_f32(
        events,
        f"{jet_name}.mass_unsmeared",
        events[jet_name].mass,
    )

    # The JER up/down branches start from the nominal JEC values.
    if shift_kind == "jer":
        events = set_ak_column_f32(
            events,
            f"{jet_name}.pt_jer_{active_direction}",
            events[jet_name].pt,
        )
        events = set_ak_column_f32(
            events,
            f"{jet_name}.mass_jer_{active_direction}",
            events[jet_name].mass,
        )

    # Same random-number generation as the old implementation.
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

    # Base variable map from the old implementation.
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

    # JER up/down use the same resolution as nominal in the old code.
    if shift_kind == "jer":
        jer_vals[active_direction] = jer_vals[jer_nom]

    # A JEC branch evaluates the resolution at its varied JEC pT.
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

    # JER scale factors. Preserve the old systematic and pT conventions.
    jersf = {}

    # Nominal SF is always needed.
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

    # Preserve the vectorized old implementation, but only over active branches.
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

    # Stochastic smearing: unchanged from old.
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

    # Gen matching: unchanged from old.
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

    # Keep the exact old behavior for numerical compatibility.
    smear_factors = ak.fill_none(
        smear_factors,
        0.0,
    )

    # Apply only nominal + active branch.
    for i, postfix in enumerate(active_postfixes):
        pt_name = f"pt{postfix}"
        mass_name = f"mass{postfix}"

        events = set_ak_column_f32(
            events,
            f"{jet_name}.{pt_name}",
            events[jet_name][pt_name] * smear_factors[..., i],
        )
        events = set_ak_column_f32(
            events,
            f"{jet_name}.{mass_name}",
            events[jet_name][mass_name] * smear_factors[..., i],
        )

    events = self[attach_coffea_behavior](
        events,
        collections=[jet_name],
        **kwargs,
    )

    return events


jer_horn_handling = jer.derive("jer_horn_handling", cls_dict={
    # source: https://cms-jerc.web.cern.ch/Recommendations/#note-25eta30
    "stochastic_smearing_mask": lambda self, jets: (abs(jets.eta) < 2.5) | (abs(jets.eta) > 3.0),
})


@jer.init
def jer_init(
    self: Calibrator,
    **kwargs,
) -> None:
    jec_cfg = self.get_jec_config()

    jec_sources = self.jec_uncertainty_sources
    if jec_sources is None:
        jec_sources = (
            jec_cfg.uncertainty_sources
            or []
        )
        self.jec_uncertainty_sources = (
            jec_sources
        )

    # Keep the complete list for ColumnFlow dependency/output declarations.
    # Runtime evaluation in jer() uses only the active local shift.
    self.jec_variations = sum(
        (
            [
                f"jec_{unc}_up",
                f"jec_{unc}_down",
            ]
            for unc
            in self.jec_uncertainty_sources
        ),
        [],
    )

    jet_jec_columns = {
        f"{self.jet_name}.{variable}_{variation}"
        for variable in ("pt", "mass")
        for variation in self.jec_variations
    }

    lower_first = (
        lambda s:
        s[0].lower() + s[1:]
        if s
        else s
    )
    self.gen_jet_idx_column = (
        lower_first(self.gen_jet_name)
        + "Idx"
    )

    # Retained for compatibility with code that inspects these attributes.
    self.jer_variations = [
        "nom",
        "up",
        "down",
    ]
    self.postfixes = [
        "",
        "_jer_up",
        "_jer_down",
    ] + [
        f"_{jec_var}"
        for jec_var
        in self.jec_variations
    ]

    self.uses |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.eta",
        f"{self.jet_name}.phi",
        f"{self.jet_name}.mass",
        f"{self.jet_name}.{self.gen_jet_idx_column}",
        f"{self.gen_jet_name}.pt",
        f"{self.gen_jet_name}.eta",
        f"{self.gen_jet_name}.phi",
    }

    # JEC-shifted branches are produced in memory by jec() immediately
    # before jer(). Mark them optional so ArrayFunction input checks do not
    # demand all JEC sources when only one active branch is constructed.
    self.uses |= {
        optional(column)
        for column
        in jet_jec_columns
    }

    # Nominal outputs are always produced.
    self.produces |= {
        f"{self.jet_name}.pt",
        f"{self.jet_name}.mass",
        f"{self.jet_name}.pt_unsmeared",
        f"{self.jet_name}.mass_unsmeared",
    }

    # Shifted branches only exist in the corresponding local-shift task.
    self.produces |= {
        optional(f"{self.jet_name}.pt_jer_{direction}")
        for direction in ("up", "down")
    }
    self.produces |= {
        optional(f"{self.jet_name}.mass_jer_{direction}")
        for direction in ("up", "down")
    }
    self.produces |= {
        optional(column)
        for column in jet_jec_columns
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
    Load the correct jer files using the :py:func:`from_string` method of the
    :external+correctionlib:py:class:`correctionlib.highlevel.CorrectionSet` function and apply the
    corrections as needed.

    The source files for the :external+correctionlib:py:class:`correctionlib.highlevel.CorrectionSet`
    instance are extracted with the :py:meth:`~.jer.get_jer_file`.

    Uses the member function :py:meth:`~.jer.get_jer_config` to construct the required keys, which
    are based on the following information about the JER:

    - campaign
    - version
    - jet_type

    A corresponding example snippet within the *config_inst* could like something like this:

    .. code-block:: python

        cfg.x.jer = DotDict.wrap({
            "Jet": {
                "campaign": f"Summer19UL{year2}{jerc_postfix}",
                "version": "JRV3",
                "jet_type": "AK4PFchs",
            },
        })

    :param reqs: Requirement dictionary for this :py:class:`~columnflow.calibration.Calibrator`
        instance.
    :param inputs: Additional inputs, currently not used.
    :param reader_targets: TODO: add documentation.
    """
    # import the correction sets from the external file
    jer_file = self.get_jer_file(reqs["external_files"].files)
    correction_set = load_correction_set(jer_file)

    # compute JER keys from config information
    jer_cfg = self.get_jer_config()
    jer_keys = {
        "jer": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_PtResolution_{jer_cfg.jet_type}",
        "sf": f"{jer_cfg.campaign}_{jer_cfg.version}_MC_ScaleFactor_{jer_cfg.jet_type}",
    }

    # store the evaluators
    self.evaluators = {
        name: get_evaluators(correction_set, [key])[0]
        for name, key in jer_keys.items()
    }

    # use deterministic seeds for random smearing if requested
    if self.deterministic_seed_index >= 0:
        idx = self.deterministic_seed_index
        bit_generator = np.random.SFC64

        def deterministic_normal(loc, scale, seed):
            return np.asarray([
                np.random.Generator(bit_generator(_seed)).normal(_loc, _scale, size=idx + 1)[-1]
                for _loc, _scale, _seed in zip(loc, scale, seed)
            ])
        self.deterministic_normal = deterministic_normal


# explicit calibrators for standard jet collections
jer_ak4 = jer.derive("jer_ak4", cls_dict={"jet_name": "Jet", "gen_jet_name": "GenJet"})

#
# single calibrator for doing both JEC and JER smearing
#

@calibrator(
    # name of the jet collection to smear
    jet_name="Jet",
    # name of the associated gen jet collection (for JER smearing)
    gen_jet_name="GenJet",
    # functions to determine configs and files
    get_jec_file=None,
    get_jec_config=None,
    get_jer_file=None,
    get_jer_config=None,
)
def jets(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Instance of :py:class:`~columnflow.calibration.Calibrator` that does all relevant calibrations
    for jets, i.e. JEC and JER. For more information, see :py:func:`~.jec` and :py:func:`~.jer`.

    :param events: awkward array containing events to process
    """
    # apply jet energy corrections
    events = self[self.jec_cls](events, **kwargs)

    # apply jer smearing on MC only
    if self.dataset_inst.is_mc:
        events = self[self.jer_cls](events, **kwargs)

    return events


@jets.init
def jets_init(self: Calibrator, **kwargs) -> None:
    # create custom jec and jer calibrators, using the jet name as the identifying value
    def get_attrs(attrs):
        cls_dict = {}
        for attr in attrs:
            if (value := getattr(self, attr, UNSET)) is not UNSET:
                cls_dict[attr] = value
        return cls_dict

    jec_attrs = ["jet_name", "gen_jet_name", "get_jec_file", "get_jec_config"]
    self.jec_cls = jec.derive(f"jec_{self.jet_name}", cls_dict=get_attrs(jec_attrs))
    self.uses.add(self.jec_cls)
    self.produces.add(self.jec_cls)

    if self.dataset_inst.is_mc:
        jer_attrs = ["jet_name", "gen_jet_name", "get_jer_file", "get_jer_config"]
        self.jer_cls = jer.derive(f"jer_{self.jet_name}", cls_dict=get_attrs(jer_attrs))
        self.uses.add(self.jer_cls)
        self.produces.add(self.jer_cls)


# explicit calibrators for standard jet collections
jets_ak4 = jets.derive("jets_ak4", cls_dict={"jet_name": "Jet", "gen_jet_name": "GenJet"})
