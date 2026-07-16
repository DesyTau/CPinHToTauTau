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

@producer(
    uses={
        "PuppiMET.pt",
        "PuppiMET.phi",
        "RecoilCorrMET.pt",
        "RecoilCorrMET.phi",
    },
    produces={
        "RecoilCorrMET.pt_unclustered_up",
        "RecoilCorrMET.phi_unclustered_up",
        "RecoilCorrMET.pt_unclustered_down",
        "RecoilCorrMET.phi_unclustered_down",
    },
)
def add_unclustered_to_recoilcorrmet( sef: Producer,
    events: ak.Array,
    **kwargs,
    ) -> ak.Array:
    """
    Build RecoilCorrMET unclustered variations by adding the NanoAOD
    unclustered MET delta to the nominal recoil-corrected MET vector.

    This keeps RecoilCorrMET as the baseline while propagating the
    unclustered MET up/down variation.
    """

    recoil_px = events.RecoilCorrMET.pt * np.cos(events.RecoilCorrMET.phi)
    recoil_py = events.RecoilCorrMET.pt * np.sin(events.RecoilCorrMET.phi)

    puppi_nom_px = events.PuppiMET.pt * np.cos(events.PuppiMET.phi)
    puppi_nom_py = events.PuppiMET.pt * np.sin(events.PuppiMET.phi)

    for direction in ("up", "down"):
        puppi_pt_name = f"pt_unclustered_{direction}"
        puppi_phi_name = f"phi_unclustered_{direction}"

        if not has_ak_column(events, f"PuppiMET.{puppi_pt_name}"):
            continue

        puppi_pt_var = events.PuppiMET[puppi_pt_name]

        if has_ak_column(events, f"PuppiMET.{puppi_phi_name}"):
            puppi_phi_var = events.PuppiMET[puppi_phi_name]
        else:
            puppi_phi_var = events.PuppiMET.phi

        puppi_var_px = puppi_pt_var * np.cos(puppi_phi_var)
        puppi_var_py = puppi_pt_var * np.sin(puppi_phi_var)

        delta_px = puppi_var_px - puppi_nom_px
        delta_py = puppi_var_py - puppi_nom_py

        recoil_var_px = recoil_px + delta_px
        recoil_var_py = recoil_py + delta_py

        recoil_var_pt = np.sqrt(recoil_var_px**2 + recoil_var_py**2)
        recoil_var_phi = np.arctan2(recoil_var_py, recoil_var_px)

        events = set_ak_column(
            events,
            f"RecoilCorrMET.pt_unclustered_{direction}",
            recoil_var_pt,
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            f"RecoilCorrMET.phi_unclustered_{direction}",
            recoil_var_phi,
            value_type=np.float32,
        )

    return events