import re

from columnflow.production.categories import category_ids as cf_category_ids

# Important:
# register all MSSM_H_tt categorizers before constructing category_ids producers
import MSSM_H_tt.categorization.main  # noqa: F401

from MSSM_H_tt.config.mass_points import (
    get_bdt_mass_blocks,
    get_bdt_mass_block_tag,
    get_bdt_masses_for_dataset,
)

_BDT_CATEGORY_MASS_RE = re.compile(r"__bdt_.+_M([0-9]+)$")


def _skip_category_outside_mass_block(
    self,
    category_inst,
):
    match = _BDT_CATEGORY_MASS_RE.search(
        category_inst.name
    )

    if not match:
        return False

    mass = int(
        match.group(1)
    )

    active_masses = (
        get_bdt_masses_for_dataset(
            self.dataset_inst,
            self.mass_points,
        )
    )

    return mass not in active_masses


CATEGORY_IDS_BLOCK_PRODUCERS = {}

for block in get_bdt_mass_blocks():
    block = tuple(block)
    tag = get_bdt_mass_block_tag(block)

    cls_name = f"category_ids_{tag}"

    producer_cls = cf_category_ids.derive(
        cls_name,
        cls_dict={
            "mass_points": block,
            "skip_category": _skip_category_outside_mass_block,
        },
    )

    globals()[cls_name] = producer_cls
    CATEGORY_IDS_BLOCK_PRODUCERS[block] = producer_cls