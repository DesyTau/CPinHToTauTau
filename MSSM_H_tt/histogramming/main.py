# coding: utf-8

"""
Histogram production and event-weight handling for MSSM H->tautau.
"""

import re

import law
import order as od

from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.hist_util import (
    create_hist_from_variables,
    fill_hist,
    translate_hist_intcat_to_strcat,
)
from columnflow.columnar_util import (
    Route,
    ak_concatenate_safe,
)
from columnflow.util import (
    maybe_import,
    pattern_matcher,
)
from columnflow.types import Any

from MSSM_H_tt.config.mass_points import (
    read_bdt_masses,
    get_bdt_masses_for_dataset,
)


np = maybe_import("numpy")
ak = maybe_import("awkward")


# -------------------------------------------------------------------------
# BDT datacard variables
# -------------------------------------------------------------------------

_BDT_CARD_VARIABLE_RE = re.compile(
    r"^bdt_"
    r"(D_sig_vs_Disc_ggphi|"
    r"D_sig_vs_Disc_bbphi|"
    r"D_DY|"
    r"D_TT)"
    r"_M[0-9]+$"
)


def _category_uses_variables(
    category: od.Category,
    variables: list[od.Variable],
) -> bool:
    """
    Restrict final BDT datacard variables to categories that explicitly
    declare them in aux["fit_var"].

    Non-BDT/datacard variables retain the normal behavior.
    """

    card_variables = {
        variable.name
        for variable in variables
        if _BDT_CARD_VARIABLE_RE.match(variable.name)
    }

    if not card_variables:
        return True

    fit_variables = category.aux.get(
        "fit_var",
        [],
    )

    if isinstance(fit_variables, str):
        fit_variables = [
            fit_variables,
        ]

    return card_variables.issubset(
        set(fit_variables)
    )


def _uses_bdt_card_variable(
    variables: list[od.Variable],
) -> bool:
    """
    Return whether at least one requested variable is a final
    mass-dependent BDT datacard variable.
    """

    return any(
        _BDT_CARD_VARIABLE_RE.match(
            variable.name
        )
        for variable in variables
    )


# -------------------------------------------------------------------------
# Dataset-dependent weights
# -------------------------------------------------------------------------

def _skip_weight_for_dataset(
    self: HistProducer,
    weight_name: str,
) -> bool:
    """
    Skip nominal weight components that do not apply to the current dataset.
    """

    dataset_name = getattr(
        self.dataset_inst,
        "name",
        "",
    )

    if (
        weight_name == "top_pt_weight"
        and not self.dataset_inst.has_tag("ttbar")
    ):
        return True

    if (
        weight_name == "stitching_weight"
        and dataset_name not in set(
            self.config_inst.x.stitch_samples
        )
    ):
        return True

    return False


# -------------------------------------------------------------------------
# Category handling
# -------------------------------------------------------------------------

def _get_bdt_category_ids(
    self: HistProducer,
    variables: list[od.Variable],
) -> tuple[int, ...]:
    """
    Return leaf-category ids that explicitly use the requested
    BDT datacard variable(s).

    Example:

        bdt_D_DY_M100

    keeps only the M100 DY categories in SR and the corresponding
    ABCD regions.
    """

    variable_key = tuple(
        variable.name
        for variable in variables
    )

    cache_key = (
        "bdt",
        variable_key,
    )

    if cache_key in self._category_cache:
        return self._category_cache[
            cache_key
        ]

    category_ids = tuple(
        category.id
        for category
        in self.config_inst.get_leaf_categories()
        if _category_uses_variables(
            category,
            variables,
        )
    )

    if not category_ids:
        raise RuntimeError(
            "no categories found for BDT variable(s): "
            + ", ".join(variable_key)
        )

    self._category_cache[
        cache_key
    ] = category_ids

    return category_ids


def _get_ordinary_category_projection(
    self: HistProducer,
):
    """
    Build the category projection used for ordinary, non-BDT variables.

    The BDT child categories are projected back onto their original
    parent analysis regions using one reference BDT mass.

    This prevents an ordinary variable such as emu_mt_tot from counting
    the same event once for every BDT mass hypothesis.

    Background/data:
        use the first active BDT mass as the reference partition.

    MSSM signal:
        get_bdt_masses_for_dataset restricts the active masses to the
        signal's own mass, so that mass is used as reference.
    """

    if self._ordinary_category_projection is not None:
        return self._ordinary_category_projection

    active_masses = get_bdt_masses_for_dataset(
        self.dataset_inst,
        read_bdt_masses(),
    )

    if not active_masses:
        raise RuntimeError(
            "no active BDT masses found for dataset "
            f"'{self.dataset_inst.name}'"
        )

    reference_mass = int(
        active_masses[0]
    )

    mass_suffix = (
        f"_M{reference_mass}"
    )

    passthrough_category_ids = []

    # Mapping:
    #
    #   parent category id
    #       -> BDT child ids at the reference mass
    #
    bdt_children_by_parent = {}

    for category in (
        self.config_inst.get_leaf_categories()
    ):

        category_name = category.name

        # Ordinary leaf category that does not belong to
        # the mass-dependent BDT hierarchy.
        if "__bdt_" not in category_name:

            passthrough_category_ids.append(
                category.id
            )

            continue

        # For ordinary variables only use one BDT partition.
        if not category_name.endswith(
            mass_suffix
        ):
            continue

        parent_name = category_name.split(
            "__bdt_",
            1,
        )[0]

        parent = self.config_inst.get_category(
            parent_name
        )

        bdt_children_by_parent.setdefault(
            parent.id,
            [],
        ).append(
            category.id
        )

    projection = {
        "reference_mass": reference_mass,

        "passthrough": tuple(
            passthrough_category_ids
        ),

        "bdt_parents": {
            parent_id: tuple(child_ids)
            for parent_id, child_ids
            in bdt_children_by_parent.items()
        },
    }

    self._ordinary_category_projection = (
        projection
    )

    return projection


def _project_ordinary_category_ids(
    self: HistProducer,
    category_ids,
):
    """
    Convert mass-dependent BDT leaf categories into a unique
    parent-category representation for ordinary variables.

    Example:

        cat_emu_sr__bdt_ggphi_and_bbphi_M60
        cat_emu_sr__bdt_dy_M60
        cat_emu_sr__bdt_tt_M60

    are collectively projected onto:

        cat_emu_sr

    Each parent category is inserted at most once per event.
    """

    projection = (
        _get_ordinary_category_projection(
            self
        )
    )

    pieces = []

    def append_category_presence(
        source_ids,
        target_id,
    ):
        """
        Add target_id once for each event belonging to at least
        one category in source_ids.
        """

        mask = np.zeros(
            len(category_ids),
            dtype=np.bool_,
        )

        for source_id in source_ids:

            mask = (
                mask
                | ak.to_numpy(
                    ak.any(
                        category_ids
                        == source_id,
                        axis=1,
                    )
                )
            )

        target_values = ak.Array(
            np.full(
                len(category_ids),
                target_id,
                dtype=np.int64,
            )
        )

        target_values = ak.mask(
            target_values,
            mask,
        )

        pieces.append(
            ak.singletons(
                target_values
            )
        )

    # Preserve ordinary categories that are already leaves.
    for category_id in (
        projection["passthrough"]
    ):

        append_category_presence(
            (category_id,),
            category_id,
        )

    # Collapse the BDT regions of the reference mass
    # onto their common parent analysis region.
    for (
        parent_id,
        child_ids,
    ) in projection[
        "bdt_parents"
    ].items():

        append_category_presence(
            child_ids,
            parent_id,
        )

    if not pieces:
        return category_ids[
            :,
            :0,
        ]

    return ak_concatenate_safe(
        pieces,
        axis=1,
    )


# -------------------------------------------------------------------------
# Histogram event weights
# -------------------------------------------------------------------------

@cf_default.hist_producer(
    keep_weights=None,
    drop_weights={
        "normalization_weight_inclusive",
    },
)
def httcp_hist_producer(
    self: HistProducer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Build nominal and embedded weight-shift event weights.

    Nominal weight components are read only once.

    For variations replacing a single component, prefix/suffix products
    avoid rebuilding the complete event weight for every variation.

    Weights are returned as an Awkward record array so that the standard
    ColumnFlow variable-selection masking remains valid.
    """

    columns = tuple(
        self.weight_columns
    )

    unit_weight = ak.Array(
        np.ones(
            len(events),
            dtype=np.float32,
        )
    )

    # ------------------------------------------------------------------
    # Read nominal weight components once
    # ------------------------------------------------------------------

    weight_components = {
        column: Route(
            column
        ).apply(
            events
        )
        for column in columns
    }

    # ------------------------------------------------------------------
    # Prefix products
    #
    # prefix[i] =
    #     product of all nominal components before component i
    # ------------------------------------------------------------------

    prefix = [
        unit_weight,
    ]

    for column in columns:

        prefix.append(
            prefix[-1]
            * weight_components[
                column
            ]
        )

    # ------------------------------------------------------------------
    # Suffix products
    #
    # suffix[i] =
    #     product beginning with component i
    # ------------------------------------------------------------------

    suffix = [
        None
    ] * (
        len(columns) + 1
    )

    suffix[
        len(columns)
    ] = unit_weight

    for index in range(
        len(columns) - 1,
        -1,
        -1,
    ):

        suffix[index] = (
            weight_components[
                columns[index]
            ]
            * suffix[
                index + 1
            ]
        )

    # ------------------------------------------------------------------
    # Preserve existing ggphi weight protection
    # ------------------------------------------------------------------

    def finalize_weight(
        weight,
    ):
        if (
            len(events)
            and
            "ggphi_phitt"
            in self.dataset_inst.name
        ):

            weight = ak.where(
                weight < 10,
                weight,
                ak.mean(weight),
            )

        return weight

    # ------------------------------------------------------------------
    # Nominal event weight
    # ------------------------------------------------------------------

    nominal_weight = finalize_weight(
        prefix[-1]
    )

    weight_dict = {
        "nominal": nominal_weight,
    }

    # ------------------------------------------------------------------
    # Embedded weight-only shifts
    #
    # Only the nominal kinematic task contains these additional shift
    # bins. JEC/JER/MET/recoil shifted tasks contain only their own
    # kinematic shift with nominal event weights.
    # ------------------------------------------------------------------

    if (
        self.dataset_inst.is_mc
        and
        task.global_shift_inst.name
        == "nominal"
    ):

        column_indices = {
            column: index
            for index, column
            in enumerate(columns)
        }

        for (
            shift_name,
            replacements,
        ) in (
            self
            .embedded_weight_shift_columns
            .items()
        ):

            # ----------------------------------------------------------
            # Variation does not apply to this dataset.
            #
            # Example:
            #
            #   top_pt_weight_up/down on a non-ttbar dataset.
            #
            # Keep an explicit shift bin identical to nominal.
            # ----------------------------------------------------------

            if not replacements:

                weight_dict[
                    shift_name
                ] = nominal_weight

                continue

            # ----------------------------------------------------------
            # Normal case:
            #
            # one nominal component is replaced by its shifted version.
            # ----------------------------------------------------------

            if len(replacements) == 1:

                (
                    nominal_column,
                    shifted_column,
                ) = next(
                    iter(
                        replacements.items()
                    )
                )

                if (
                    nominal_column
                    not in column_indices
                ):

                    raise RuntimeError(
                        f"cannot replace weight "
                        f"'{nominal_column}' for "
                        f"shift '{shift_name}': "
                        "nominal weight is not active "
                        "for this dataset"
                    )

                index = column_indices[
                    nominal_column
                ]

                shifted_component = Route(
                    shifted_column
                ).apply(
                    events
                )

                shifted_weight = (
                    prefix[index]
                    * shifted_component
                    * suffix[
                        index + 1
                    ]
                )

            else:

                # ------------------------------------------------------
                # Generic fallback in case a future systematic modifies
                # multiple nominal weight components simultaneously.
                # ------------------------------------------------------

                shifted_weight = (
                    unit_weight
                )

                for column in columns:

                    shifted_column = (
                        replacements.get(
                            column,
                            None,
                        )
                    )

                    if shifted_column is None:

                        component = (
                            weight_components[
                                column
                            ]
                        )

                    else:

                        component = Route(
                            shifted_column
                        ).apply(
                            events
                        )

                    shifted_weight = (
                        shifted_weight
                        * component
                    )

            weight_dict[
                shift_name
            ] = finalize_weight(
                shifted_weight
            )

    # ------------------------------------------------------------------
    # Important:
    #
    # return an Awkward record instead of a Python dictionary.
    #
    # This allows ColumnFlow to perform:
    #
    #     masked_weights = masked_weights[mask]
    #
    # for variables with callable selections.
    # ------------------------------------------------------------------

    return events, ak.zip(
        weight_dict,
        depth_limit=1,
    )


# -------------------------------------------------------------------------
# Histogram producer initialization
# -------------------------------------------------------------------------

@httcp_hist_producer.init
def httcp_hist_init(
    self: HistProducer,
) -> None:

    # Category lookup caches.
    self._category_cache = {}

    self._ordinary_category_projection = (
        None
    )

    # Nominal event-weight components.
    self.weight_columns = []

    # Maps, for example:
    #
    #   muon_weight_up:
    #       {
    #           "muon_weight":
    #               "muon_weight_up"
    #       }
    #
    self.embedded_weight_shift_columns = {}

    do_keep = (
        pattern_matcher(
            self.keep_weights
        )
        if self.keep_weights
        else lambda _, /: True
    )

    do_drop = (
        pattern_matcher(
            self.drop_weights
        )
        if self.drop_weights
        else lambda _, /: False
    )

    # Data has unit event weights.
    if self.dataset_inst.is_data:
        return

    all_weights = (
        self.config_inst
        .x.event_weights
        .copy()
    )

    all_weights.update(
        self.dataset_inst.x(
            "event_weights",
            {},
        )
    )

    embedded_sources = set(
        self.config_inst.x(
            "histogram_weight_shift_sources",
            (),
        )
    )

    for (
        weight_name,
        shift_insts,
    ) in all_weights.items():

        if (
            not do_keep(
                weight_name
            )
            or
            do_drop(
                weight_name
            )
        ):
            continue

        skip_weight = (
            _skip_weight_for_dataset(
                self,
                weight_name,
            )
        )

        # --------------------------------------------------------------
        # Nominal weight component
        # --------------------------------------------------------------

        if not skip_weight:

            self.weight_columns.append(
                weight_name
            )

            self.uses.add(
                weight_name
            )

        # --------------------------------------------------------------
        # Register all shifts implemented by this weight
        # --------------------------------------------------------------

        self.shifts |= {
            shift_inst.name
            for shift_inst
            in shift_insts
        }

        # --------------------------------------------------------------
        # Determine which weight-only shifts are embedded directly
        # into the nominal histogram.
        # --------------------------------------------------------------

        for shift_inst in shift_insts:

            if (
                shift_inst.source
                not in embedded_sources
            ):
                continue

            replacements = {}

            # If the nominal weight does not apply to the current
            # dataset, the corresponding up/down histograms will be
            # identical to nominal.
            if not skip_weight:

                aliases = shift_inst.x(
                    "column_aliases",
                    {},
                )

                shifted_column = (
                    aliases.get(
                        weight_name
                    )
                )

                if shifted_column is None:

                    raise RuntimeError(
                        "no shifted column alias found "
                        f"for weight '{weight_name}' "
                        f"and shift "
                        f"'{shift_inst.name}'"
                    )

                replacements[
                    weight_name
                ] = shifted_column

                self.uses.add(
                    shifted_column
                )

            self.embedded_weight_shift_columns[
                shift_inst.name
            ] = replacements


# -------------------------------------------------------------------------
# Histogram creation
# -------------------------------------------------------------------------

@httcp_hist_producer.create_hist
def httcp_create_hist(
    self: HistProducer,
    variables: list[od.Variable],
    task: law.Task,
    **kwargs,
):
    """
    Create one histogram for the requested variable tuple.

    Categories, processes and shifts are represented by histogram axes.
    No separate histogram object is created for each category.
    """

    return create_hist_from_variables(
        *variables,
        categorical_axes=(
            (
                "category",
                "intcat",
            ),
            (
                "process",
                "intcat",
            ),
            (
                "shift",
                "intcat",
            ),
        ),
        weight=True,
    )


# -------------------------------------------------------------------------
# Histogram filling
# -------------------------------------------------------------------------

@httcp_hist_producer.fill_hist
def httcp_fill_hist(
    self: HistProducer,
    h,
    data: dict[str, Any],
    variables: list[od.Variable],
    events: ak.Array,
    task: law.Task,
) -> None:
    """
    Fill one histogram containing category, process and shift axes.

    BDT datacard variables:
        retain only the explicitly associated mass/BDT categories.

    Ordinary variables:
        project the mass-dependent BDT leaf categories back onto the
        original parent analysis categories, preventing multiple counting
        across BDT mass hypotheses.
    """

    # ------------------------------------------------------------------
    # Determine the categories to fill
    # ------------------------------------------------------------------

    if _uses_bdt_card_variable(
        variables
    ):

        relevant_category_ids = (
            _get_bdt_category_ids(
                self,
                variables,
            )
        )

        category_mask = (
            ak.zeros_like(
                data["category"],
                dtype=np.bool_,
            )
        )

        for category_id in (
            relevant_category_ids
        ):

            category_mask = (
                category_mask
                | (
                    data["category"]
                    == category_id
                )
            )

        categories_to_fill = (
            data["category"][
                category_mask
            ]
        )

    else:

        categories_to_fill = (
            _project_ordinary_category_ids(
                self,
                data["category"],
            )
        )

    # ------------------------------------------------------------------
    # Event weights
    #
    # data["weight"] is the Awkward record produced by
    # httcp_hist_producer.
    # ------------------------------------------------------------------

    weight_fields = ak.fields(
        data["weight"]
    )

    if not weight_fields:

        raise RuntimeError(
            "histogram weight record "
            "contains no fields"
        )

    # ------------------------------------------------------------------
    # Fill nominal + embedded weight shifts
    # ------------------------------------------------------------------

    for weight_shift_name in (
        weight_fields
    ):

        event_weight = (
            data["weight"][
                weight_shift_name
            ]
        )

        if (
            weight_shift_name
            == "nominal"
        ):

            # For nominal tasks:
            #
            #   nominal shift id
            #
            # For JEC/JER/MET/recoil tasks:
            #
            #   corresponding kinematic-shift id
            #
            shift_id = (
                data["shift"]
            )

        else:

            # Embedded weight-only systematic.
            shift_id = (
                self.config_inst
                .get_shift(
                    weight_shift_name
                )
                .id
            )

        fill_data = {
            "category":
                categories_to_fill,

            "process":
                data["process"],

            "shift":
                shift_id,

            "weight":
                event_weight,
        }

        for variable_inst in (
            variables
        ):

            fill_data[
                variable_inst.name
            ] = data[
                variable_inst.name
            ]

        fill_hist(
            h,
            fill_data,
            last_edge_inclusive=(
                task.last_edge_inclusive
            ),
        )


# -------------------------------------------------------------------------
# Histogram post-processing
# -------------------------------------------------------------------------

@httcp_hist_producer.post_process_hist
def httcp_post_process_hist(
    self: HistProducer,
    h,
    task: law.Task,
) -> Any:
    """
    Convert integer category/process/shift axes to string axes.

    This makes histograms from different configurations compatible and
    allows downstream plotting/inference code to select objects by name.
    """

    axis_names = {
        axis.name
        for axis in h.axes
    }

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    if "process" in axis_names:

        process_map = {
            int(process_id):
                self.config_inst
                .get_process(
                    int(process_id)
                )
                .name

            for process_id
            in h.axes["process"]
        }

        h = (
            translate_hist_intcat_to_strcat(
                h,
                "process",
                process_map,
            )
        )

    # ------------------------------------------------------------------
    # Shift
    # ------------------------------------------------------------------

    if "shift" in axis_names:

        shift_map = {
            int(shift_id):
                self.config_inst
                .get_shift(
                    int(shift_id)
                )
                .name

            for shift_id
            in h.axes["shift"]
        }

        h = (
            translate_hist_intcat_to_strcat(
                h,
                "shift",
                shift_map,
            )
        )

    # ------------------------------------------------------------------
    # Category
    # ------------------------------------------------------------------

    if "category" in axis_names:

        category_map = {
            int(category_id):
                self.config_inst
                .get_category(
                    int(category_id)
                )
                .name

            for category_id
            in h.axes["category"]
        }

        h = (
            translate_hist_intcat_to_strcat(
                h,
                "category",
                category_map,
            )
        )

    return h