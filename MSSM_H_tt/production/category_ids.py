import re

from columnflow.production import Producer
from columnflow.production.categories import category_ids as cf_category_ids

# Important:
# register all MSSM_H_tt categorizers before constructing category_ids producers
import MSSM_H_tt.categorization.main  # noqa: F401

from MSSM_H_tt.config.mass_points import (
    get_bdt_mass_blocks,
    get_bdt_mass_block_tag,
    get_bdt_masses_for_dataset,
)


_BDT_CATEGORY_MASS_RE = re.compile(
    r"__bdt_"
    r"(?:ggphi_and_bbphi|dy|tt)"
    r"_M([0-9]+)$"
)


def _skip_bdt_category_for_dataset(
    self: Producer,
    category_inst,
) -> bool:
    """
    Skip BDT categories whose mass is not evaluated for the
    current dataset.

    Backgrounds and data:
        keep all configured BDT masses.

    MSSM signals:
        keep only the BDT categories corresponding to the
        signal mass.
    """

    match = _BDT_CATEGORY_MASS_RE.search(
        category_inst.name
    )

    # Ordinary non-BDT categories are always kept.
    if not match:
        return False

    category_mass = int(
        match.group(1)
    )

    active_masses = get_bdt_masses_for_dataset(
        self.dataset_inst
    )

    return category_mass not in active_masses


# -------------------------------------------------------------------------
# Full category_ids producer
#
# Used by the generic "main" producer.
#
# Background/data:
#     all BDT masses
#
# Signal:
#     only its own BDT mass
# -------------------------------------------------------------------------

mssm_category_ids = cf_category_ids.derive(
    "mssm_category_ids",
    cls_dict={
        "skip_category":
            _skip_bdt_category_for_dataset,
    },
)


# -------------------------------------------------------------------------
# Block-aware category_ids producers
#
# Used by bdt_card.py.
# -------------------------------------------------------------------------

def _skip_category_outside_mass_block(
    self: Producer,
    category_inst,
) -> bool:

    match = _BDT_CATEGORY_MASS_RE.search(
        category_inst.name
    )

    # Ordinary non-BDT categories are always kept.
    if not match:
        return False

    category_mass = int(
        match.group(1)
    )

    active_masses = get_bdt_masses_for_dataset(
        self.dataset_inst,
        self.mass_points,
    )

    return category_mass not in active_masses


CATEGORY_IDS_BLOCK_PRODUCERS = {}


for block in get_bdt_mass_blocks():

    block = tuple(
        block
    )

    tag = get_bdt_mass_block_tag(
        block
    )

    cls_name = (
        f"category_ids_{tag}"
    )

    producer_cls = cf_category_ids.derive(
        cls_name,
        cls_dict={
            "mass_points":
                block,
            "skip_category":
                _skip_category_outside_mass_block,
        },
    )

    globals()[
        cls_name
    ] = producer_cls

    CATEGORY_IDS_BLOCK_PRODUCERS[
        block
    ] = producer_cls