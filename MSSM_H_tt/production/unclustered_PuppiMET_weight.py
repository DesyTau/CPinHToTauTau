# coding: utf-8

"""
Producer to build unclustered MET weights.
"""

from __future__ import annotations

import functools

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "PuppiMET.pt",
        "PuppiMET.ptUnclusteredUp",
        "PuppiMET.ptUnclusteredDown",
    },
    produces={
        "Unclustered_weight",
        "Unclustered_weight_up",
        "Unclustered_weight_down",
    },
)
def unclustered_weight(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    puppimet_pt = events.PuppiMET.pt
    valid = puppimet_pt != 0

    Unclustered_weight = ak.where(valid, puppimet_pt / puppimet_pt, 1.0)
    Unclustered_weight_up = ak.where(valid, events.PuppiMET.ptUnclusteredUp / puppimet_pt, 1.0)
    Unclustered_weight_down = ak.where(valid, events.PuppiMET.ptUnclusteredDown / puppimet_pt, 1.0)

    events = set_ak_column_f32(events, "Unclustered_weight", Unclustered_weight)
    events = set_ak_column_f32(events, "Unclustered_weight_up", Unclustered_weight_up)
    events = set_ak_column_f32(events, "Unclustered_weight_down", Unclustered_weight_down)

    return events