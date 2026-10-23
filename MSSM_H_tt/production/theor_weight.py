import functools

from columnflow.production import Producer, producer
from columnflow.columnar_util import (
    set_ak_column,
    has_ak_column,
    flat_np_view,
    optional_column as optional,
)
from columnflow.util import maybe_import, load_correction_set

import law

logger = law.logger.get_logger(__name__)

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
cl = maybe_import("correctionlib")
warn = maybe_import("warnings")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={"event", optional("LHEScaleWeight.*"), optional("PSWeight.*")},
    produces={"lhe_weight*", "ps_weight*"},
    mc_only=True,
)
def theor_unc(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    unit_weight = ak.ones_like(events.event, dtype=np.float32)

    dataset_inst = getattr(self, "dataset_inst", None)

    is_dy_or_ttbar = (
        dataset_inst is not None
        and (
            dataset_inst.has_tag("dy")
            or dataset_inst.has_tag("ttbar")
        )
    )
    # -------------------------------------------------------------------------
    # Audit availability of theory-variation columns.
    #
    # Print only once per producer instance / dataset.
    # -------------------------------------------------------------------------

    if not getattr(self, "_theory_audit_printed", False):
        dataset_name = (
            dataset_inst.name
            if dataset_inst is not None
            else "<unknown>"
        )

        has_lhe_field = "LHEScaleWeight" in events.fields
        has_ps_field = "PSWeight" in events.fields

        has_valid_lhe = False
        has_valid_ps = False

        if len(events):
            if has_lhe_field:
                has_valid_lhe = bool(
                    ak.any(
                        ak.num(
                            events.LHEScaleWeight,
                            axis=1,
                        ) == 9
                    )
                )

            if has_ps_field:
                has_valid_ps = bool(
                    ak.any(
                        ak.num(
                            events.PSWeight,
                            axis=1,
                        ) == 4
                    )
                )

        logger.info(
            "[THEORY AUDIT] dataset=%s | "
            "DY/ttbar_skip=%s | "
            "LHEScaleWeight=%s valid9=%s | "
            "PSWeight=%s valid4=%s",
            dataset_name,
            is_dy_or_ttbar,
            has_lhe_field,
            has_valid_lhe,
            has_ps_field,
            has_valid_ps,
        )

        self._theory_audit_printed = True
    if is_dy_or_ttbar:
        print("Skipping theoretical uncertainty production for DY/ttbar. Adding dummy weights = 1.")

        # nominal LHE and PS weights
        events = set_ak_column_f32(events, "lhe_weight", unit_weight)
        events = set_ak_column_f32(events, "ps_weight", unit_weight)

        # dummy LHE variations
        for syst_name in self.config_inst.x.lhe_variations:
            events = set_ak_column_f32(
                events,
                f"lhe_weight_{syst_name}",
                unit_weight,
            )

        # dummy PS variations
        for syst_name in self.config_inst.x.ps_variations:
            events = set_ak_column_f32(
                events,
                f"ps_weight_{syst_name}",
                unit_weight,
            )

        return events

    print("Producing theoretical uncertainties...")

    # nominal LHE weight
    events = set_ak_column_f32(events, "lhe_weight", unit_weight)

    for syst_name, bin_idx in self.config_inst.x.lhe_variations.items():

        if "LHEScaleWeight" in events.fields:
            # apply only when all 9 LHE scale weights are available
            mask = ak.num(events.LHEScaleWeight, axis=1) == 9

            the_weight = ak.fill_none(
                ak.firsts(events.LHEScaleWeight[:, bin_idx:]),
                1.0,
            )

            lhe_weight_syst = ak.where(
                mask,
                the_weight,
                unit_weight,
            )

            # protect against pathological large weights
            # if weight > 10, replace it with 1
            lhe_weight_syst = ak.where(
                lhe_weight_syst > 10.0,
                unit_weight,
                lhe_weight_syst,
            )

        else:
            lhe_weight_syst = unit_weight

        events = set_ak_column_f32(
            events,
            f"lhe_weight_{syst_name}",
            lhe_weight_syst,
        )

    # nominal PS weight
    events = set_ak_column_f32(events, "ps_weight", unit_weight)

    for syst_name, bin_idx in self.config_inst.x.ps_variations.items():

        if "PSWeight" in events.fields:
            # apply only when all 4 PS weights are available
            mask = ak.num(events.PSWeight, axis=1) == 4

            the_weight = ak.fill_none(
                ak.firsts(events.PSWeight[:, bin_idx:]),
                1.0,
            )

            ps_weight_syst = ak.where(
                mask,
                the_weight,
                unit_weight,
            )

            # protect against pathological large weights
            # if weight > 10, replace it with 1
            ps_weight_syst = ak.where(
                ps_weight_syst > 10.0,
                unit_weight,
                ps_weight_syst,
            )

        else:
            ps_weight_syst = unit_weight

        events = set_ak_column_f32(
            events,
            f"ps_weight_{syst_name}",
            ps_weight_syst,
        )

    return events