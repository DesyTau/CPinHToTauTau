# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /afs/cern.ch/work/a/anigamov/DesyTau/CPinHToTauTau/httcp/histogramming/main.py
# Bytecode version: 3.9.0beta5 (3425)
# Source timestamp: 2025-07-22 15:21:18 UTC (1753197678)

"""
Example event weight producer.
"""
import law
import order as od
from columnflow.histogramming import HistProducer, hist_producer
from columnflow.histogramming.default import cf_default, create_hist_from_variables, fill_hist, translate_hist_intcat_to_strcat
from columnflow.hist_util import add_hist_axis
from columnflow.columnar_util import Route
from columnflow.util import maybe_import, pattern_matcher
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.types import Any
import warnings
import re
np = maybe_import('numpy')
ak = maybe_import('awkward')
hist = maybe_import('hist')

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
    declare them in aux['fit_var'].

    Non-BDT/datacard variables retain the previous behavior.
    """

    card_variables = {
        variable.name
        for variable in variables
        if _BDT_CARD_VARIABLE_RE.match(
            variable.name
        )
    }

    # Preserve the old behavior for all ordinary/diagnostic variables.
    if not card_variables:
        return True

    fit_variables = category.aux.get(
        "fit_var",
        [],
    )

    if isinstance(
        fit_variables,
        str,
    ):
        fit_variables = [
            fit_variables,
        ]

    fit_variables = set(
        fit_variables
    )

    return card_variables.issubset(
        fit_variables
    )
    
def _skip_weight_for_dataset(
    self: HistProducer,
    weight_name: str,
) -> bool:

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
@cf_default.hist_producer(
    keep_weights=None,
    skip_compatibility_check=True,
    drop_weights={"normalization_weight_inclusive"},
)
def httcp_hist_producer(
    self: HistProducer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> ak.Array:

    def build_weight(replacements=None):

        replacements = replacements or {}

        weight = ak.Array(
            np.ones(
                len(events),
                dtype=np.float32,
            )
        )

        for column in self.weight_columns:

            route = replacements.get(
                column,
                column,
            )

            weight = (
                weight
                * Route(route).apply(events)
            )

        if "ggphi_phitt" in self.dataset_inst.name:
            weight = ak.where(
                weight < 10,
                weight,
                ak.mean(weight),
            )

        return weight

    # nominal event weight
    nominal_weight = build_weight()

    weight_dict = {
        "nominal": nominal_weight,
    }

    # Only the nominal kinematic task embeds all weight variations.
    #
    # For JEC/JER/MET/etc. tasks, continue filling just that
    # particular kinematic shift with nominal event weights.
    if (
        self.dataset_inst.is_mc
        and task.global_shift_inst.name == "nominal"
    ):

        for (
            shift_name,
            replacements,
        ) in self.embedded_weight_shift_columns.items():

            weight_dict[shift_name] = build_weight(
                replacements
            )

    return events, weight_dict

@httcp_hist_producer.init
def httcp_hist_init(
    self: HistProducer,
) -> None:

    self.weight_columns = []

    # Maps:
    #
    #   muon_weight_up:
    #       {"muon_weight": "muon_weight_up"}
    #
    # etc.
    self.embedded_weight_shift_columns = {}

    do_keep = (
        pattern_matcher(self.keep_weights)
        if self.keep_weights
        else lambda _, /: True
    )

    do_drop = (
        pattern_matcher(self.drop_weights)
        if self.drop_weights
        else lambda _, /: False
    )

    if self.dataset_inst.is_data:
        return

    all_weights = (
        self.config_inst.x.event_weights.copy()
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
            not do_keep(weight_name)
            or do_drop(weight_name)
        ):
            continue

        skip_weight = _skip_weight_for_dataset(
            self,
            weight_name,
        )

        # Only nominally applied weights belong here.
        if not skip_weight:
            self.weight_columns.append(
                weight_name
            )

            self.uses.add(
                weight_name
            )

        self.shifts |= {
            shift_inst.name
            for shift_inst in shift_insts
        }

        for shift_inst in shift_insts:

            if (
                shift_inst.source
                not in embedded_sources
            ):
                continue

            replacements = {}

            # When the nominal weight itself is skipped
            # (e.g. top_pt_weight outside ttbar), retain
            # an up/down histogram identical to nominal.
            if not skip_weight:

                aliases = shift_inst.x(
                    "column_aliases",
                    {},
                )

                shifted_column = aliases.get(
                    weight_name
                )

                if shifted_column is None:
                    raise RuntimeError(
                        f"no shifted column alias found for "
                        f"weight '{weight_name}' and shift "
                        f"'{shift_inst.name}'"
                    )

                replacements[weight_name] = shifted_column

                self.uses.add(shifted_column)

            self.embedded_weight_shift_columns[shift_inst.name] = replacements

@httcp_hist_producer.create_hist
def httcp_create_hist(
    self: HistProducer,
    variables: list[od.Variable],
    task: law.Task,
    **kwargs,
) -> dict:
    """
    Define histograms only for categories that use the requested
    final BDT datacard variable.

    Ordinary variables retain the original all-category behavior.
    """

    histograms = {}

    for category in self.config_inst.categories:
        if not _category_uses_variables(
            category,
            variables,
        ):
            continue

        histograms[
            category.name
        ] = create_hist_from_variables(
            *variables,
            categorical_axes=(
                ("category", "intcat"),
                ("process", "intcat"),
                ("shift", "intcat"),
            ),
            weight=True,
        )

    return histograms

@httcp_hist_producer.fill_hist
def httcp_fill_hist(
    self: HistProducer,
    h: dict,
    data: dict[str, Any],
    variables: list[od.Variable],
    events: ak.Array,
    task: law.Task,
) -> None:

    for cat_name in self.config_inst.categories.names():

        cat = self.config_inst.get_category(
            cat_name
        )

        mask = ak.any(
            data["category"] == cat.id,
            axis=1,
        )

        # Ordinary categories:
        # fill nominal plus all embedded weight shifts.
        if "apply_ff" not in cat.aux:

            weights_to_fill = (
                data["weight"]
            )

        elif cat.aux["apply_ff"] == "wj":

            weights_to_fill = {
                "nominal": data["weight"]["tf_wj"],
            }

        elif cat.aux["apply_ff"] == "qcd":

            weights_to_fill = {
                "nominal": data["weight"]["tf_qcd"],
            }

        else:
            weights_to_fill = {
                "nominal":
                    data["weight"]["nominal"],
            }

        for (
            weight_shift_name,
            event_weight,
        ) in weights_to_fill.items():

            fill_data = {}

            masked_weight = event_weight[mask]

            fill_data["weight"] = (
                masked_weight
            )

            fill_data["category"] = (
                ak.full_like(
                    masked_weight,
                    cat.id,
                    dtype=np.int32,
                )
            )

            # "nominal" means the kinematic shift of
            # this CreateHistograms task.
            #
            # For the nominal task this is shift 0.
            # For a JEC/JER task it is that JEC/JER id.
            if weight_shift_name == "nominal":

                shift_id = data["shift"]

            else:

                shift_id = (
                    self.config_inst
                    .get_shift(
                        weight_shift_name
                    )
                    .id
                )

            fill_data["shift"] = (
                ak.full_like(
                    masked_weight,
                    shift_id,
                    dtype=np.int32,
                )
            )

            fill_data["process"] = (
                data["process"][mask]
            )

            for variable_inst in variables:

                var_name = (
                    variable_inst.name
                )

                fill_data[var_name] = (
                    data[var_name][mask]
                )

            fill_hist(
                h[cat.name],
                fill_data,
                last_edge_inclusive=(
                    task.last_edge_inclusive
                ),
            )
@httcp_hist_producer.post_process_hist
def default_post_process_hist(
    self: HistProducer,
    h: dict,
    task: law.Task,
) -> Any:
    """
    Post-process the histogram, combining the per-category histograms and
    converting integer categorical axes to string axes.
    """

    h_list = list(h.values())

    if not h_list:
        raise RuntimeError("No histograms available for post-processing")

    # Merge category histograms.
    h_merged = sum(
        h_list[1:],
        h_list[0].copy(),
    )

    axis_names = {
        ax.name
        for ax in h_merged.axes
    }

    if "process" in axis_names:
        process_map = {
            proc_id: self.config_inst.get_process(proc_id).name
            for proc_id in h_merged.axes["process"]
        }

        h_merged = translate_hist_intcat_to_strcat(
            h_merged,
            "process",
            process_map,
        )

    if "shift" in axis_names:

        shift_map = {
            int(shift_id):
                self.config_inst
                .get_shift(int(shift_id))
                .name
            for shift_id
            in h_merged.axes["shift"]
        }

        h_merged = translate_hist_intcat_to_strcat(
            h_merged,
            "shift",
            shift_map,
        )

    if "category" in axis_names:
        category_map = {
            cat_id: self.config_inst.get_category(cat_id).name
            for cat_id in h_merged.axes["category"]
        }

        h_merged = translate_hist_intcat_to_strcat(
            h_merged,
            "category",
            category_map,
        )

    return h_merged