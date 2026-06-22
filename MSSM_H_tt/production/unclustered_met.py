# coding: utf-8

"""
Producer to build unclustered MET shape variations.
"""

from __future__ import annotations

import functools

from columnflow.production import Producer, producer
from columnflow.columnar_util import (
    set_ak_column,
    has_ak_column,
    optional_column as optional,
)
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "PuppiMET.pt",
        "PuppiMET.phi",
        "PuppiMET.ptUnclusteredUp",
        "PuppiMET.ptUnclusteredDown",
        optional("PuppiMET.phiUnclusteredUp"),
        optional("PuppiMET.phiUnclusteredDown"),
    },
    produces={
        "PuppiMET.pt_unclustered_up",
        "PuppiMET.phi_unclustered_up",
        "PuppiMET.pt_unclustered_down",
        "PuppiMET.phi_unclustered_down",
    },
)
def unclustered_met(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Build unclustered MET shape variations.

    Output columns:

        PuppiMET.pt_unclustered_up
        PuppiMET.phi_unclustered_up
        PuppiMET.pt_unclustered_down
        PuppiMET.phi_unclustered_down

    These should be connected to ColumnFlow shifts via add_shift_aliases, not via
    cfg.x.event_weights.
    """

    # pt variations exist in NanoAOD.
    met_pt_up = events.PuppiMET.ptUnclusteredUp
    met_pt_down = events.PuppiMET.ptUnclusteredDown

    # phi variations should exist for proper propagation.
    # If they are missing in a given NanoAOD campaign, keep nominal phi.
    if has_ak_column(events, "PuppiMET.phiUnclusteredUp"):
        met_phi_up = events.PuppiMET.phiUnclusteredUp
    else:
        met_phi_up = events.PuppiMET.phi

    if has_ak_column(events, "PuppiMET.phiUnclusteredDown"):
        met_phi_down = events.PuppiMET.phiUnclusteredDown
    else:
        met_phi_down = events.PuppiMET.phi

    events = set_ak_column_f32(events, "PuppiMET.pt_unclustered_up", met_pt_up)
    events = set_ak_column_f32(events, "PuppiMET.phi_unclustered_up", met_phi_up)

    events = set_ak_column_f32(events, "PuppiMET.pt_unclustered_down", met_pt_down)
    events = set_ak_column_f32(events, "PuppiMET.phi_unclustered_down", met_phi_down)

    return events