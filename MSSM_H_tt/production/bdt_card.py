# coding: utf-8

from columnflow.production import (
    Producer,
    producer,
)

from MSSM_H_tt.config.mass_points import (
    get_bdt_mass_blocks,
    get_bdt_mass_block_tag,
)

from MSSM_H_tt.production.bdt_score import (
    MSSM_BDT_SCORE_BLOCK_PRODUCERS,
)

from MSSM_H_tt.production.bdt_2d_bins import (
    BDT_2D_CARD_BLOCK_PRODUCERS,
)

from MSSM_H_tt.production.category_ids import (
    CATEGORY_IDS_BLOCK_PRODUCERS,
)


BDT_CARD_VARIABLES = (
    "D_sig_vs_Disc_ggphi",
    "D_sig_vs_Disc_bbphi",
    "D_DY",
    "D_TT",
)


def _card_output_columns(masses):
    return {
        "category_ids",
    } | {
        f"bdt_{variable}_M{mass}"
        for mass in masses
        for variable in BDT_CARD_VARIABLES
    }


def _make_bdt_card_producer(block):
    block = tuple(block)
    tag = get_bdt_mass_block_tag(block)

    score_producer = (
        MSSM_BDT_SCORE_BLOCK_PRODUCERS[block]
    )
    bdt_2d_producer = (
        BDT_2D_CARD_BLOCK_PRODUCERS[block]
    )
    category_producer = (
        CATEGORY_IDS_BLOCK_PRODUCERS[block]
    )

    cls_name = f"bdt_card_{tag}"

    @producer(
        cls_name=cls_name,
        uses={
            score_producer,
            bdt_2d_producer,
            category_producer,
        },
        produces=_card_output_columns(block),

        # main_common is produced once and reused by all
        # BDT mass blocks.
        require_producers={
            "main_common",
        },

        mass_points=block,
    )
    def card_producer(
        self: Producer,
        events,
        **kwargs,
    ):
        events = self[score_producer](
            events,
            **kwargs,
        )

        events = self[bdt_2d_producer](
            events,
            **kwargs,
        )

        events = self[category_producer](
            events,
            **kwargs,
        )

        return events

    @card_producer.init
    def card_producer_init(
        self: Producer,
        **kwargs,
    ):
        # Only shifts that can alter the BDT input
        # kinematics need a new BDT evaluation.
        self.shifts |= {
            shift_inst.name
            for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag(
                ("jec", "jer", "met", "met_recoil")
            )
        }

    return card_producer


BDT_CARD_BLOCK_PRODUCERS = {}


for block in get_bdt_mass_blocks():
    block = tuple(block)

    producer_cls = _make_bdt_card_producer(
        block
    )

    globals()[producer_cls.cls_name] = (
        producer_cls
    )

    BDT_CARD_BLOCK_PRODUCERS[block] = (
        producer_cls
    )