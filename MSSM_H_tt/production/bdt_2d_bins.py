# coding: utf-8

"""
Producer for flattened 2D BDT variables used by the MSSM H->tautau e-mu BDT.

For rectangular 2D variables, the flattening convention is

    flat_index = ix * n_y_bins + iy
    stored value = flat_index + 0.5

For y-merged 2D variables produced by the training post-processing, the y-axis
binning is different in each x bin. In that case, the flattening convention is

    flat_index = x_bin_offset[ix] + iy
    stored value = flat_index + 0.5

where

    x_bin_offset[ix] = sum(n_y_bins for all previous x bins)

Invalid or out-of-range values are stored as -1.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Union

import numpy as np

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from MSSM_H_tt.config.mass_points import (
    read_bdt_masses,
    get_bdt_mass_blocks,
    get_bdt_mass_block_tag,
)


ak = maybe_import("awkward")


MASS_POINTS = tuple(read_bdt_masses())

Mass = Union[int, str]


# -------------------------------------------------------------------------
# BDT binning output location
# -------------------------------------------------------------------------

# Directory containing the BDT training/post-processing outputs. This is used
# only to read the irregular 2D y-merged binning JSONs.
#
# Can be overridden at runtime with:
#   export MSSM_BDT_2D_BINNING_BASE=/path/to/bdt/output/base
BDT_2D_BINNING_BASE = Path(
    os.environ.get(
        "MSSM_BDT_2D_BINNING_BASE",
        "/eos/project/d/desytau/public/jmalvaso/"
        "bdt_3_classes_10_features_clippedJetCounts",
    )
)

BDT_ADAPTIVE_TAG = "combined_crossApplied"
BDT_2D_BINNING_SUBDIR = "independent_adaptive"


# -------------------------------------------------------------------------
# BDT inputs and outputs
# -------------------------------------------------------------------------

# Input columns already produced by the current BDT-score producer:
#
#   bdt_D_sig_M{MASS}
#   bdt_D_ggphi_M{MASS}
#   bdt_D_bbphi_M{MASS}
#   bdt_Disc_ggphi_M{MASS}
#   bdt_Disc_bbphi_M{MASS}
#
# The old version of this producer derived Disc_ggphi and Disc_bbphi from
# D_ggphi and D_bbphi. With the new BDT-score producer, these are direct BDT
# output columns and are treated as inputs here.
BDT_INPUTS = (
    "D_sig",
    "D_ggphi",
    "D_bbphi",
    "Disc_ggphi",
    "Disc_bbphi",
)

# Full set, kept for plots/general production
BDT_2D_PAIRS = (
    ("D_sig_vs_D_ggphi", "D_sig", "D_ggphi"),
    ("D_sig_vs_D_bbphi", "D_sig", "D_bbphi"),
    ("D_ggphi_vs_D_bbphi", "D_ggphi", "D_bbphi"),

    ("D_sig_vs_Disc_ggphi", "D_sig", "Disc_ggphi"),
    ("D_sig_vs_Disc_bbphi", "D_sig", "Disc_bbphi"),
)



# Flattened 2D variables to produce.
BDT_CARD_2D_PAIRS = (
    (
        "D_sig_vs_Disc_ggphi",
        "D_sig",
        "Disc_ggphi",
    ),
    (
        "D_sig_vs_Disc_bbphi",
        "D_sig",
        "Disc_bbphi",
    ),
)


BDT_YMERGED_2D_PAIRS = {
    ("D_sig", "Disc_ggphi"),
    ("D_sig", "Disc_bbphi"),
}


# -------------------------------------------------------------------------
# Naming helpers
# -------------------------------------------------------------------------
def _bdt_2d_input_columns(
    masses,
    pairs,
):
    discriminants = {
        disc
        for _, x_disc, y_disc in pairs
        for disc in (x_disc, y_disc)
    }

    return {
        _column_name(discriminant, mass)
        for mass in masses
        for discriminant in discriminants
    }


def _bdt_2d_output_columns(
    masses,
    pairs,
):
    return {
        _pair_column_name(pair_name, mass)
        for mass in masses
        for pair_name, _, _ in pairs
    }
    
def _column_name(discriminant: str, mass: Mass) -> str:
    return f"bdt_{discriminant}_M{mass}"


def _pair_column_name(pair_name: str, mass: Mass) -> str:
    return f"bdt_{pair_name}_M{mass}"


def _safe_pair_name(x_discriminant: str, y_discriminant: str) -> str:
    # Must match adaptive_plot_2d_discriminant_pair in the training script:
    #   safe_pair = f"{y_name}_vs_{x_name}".replace("/", "_")
    return f"{y_discriminant}_vs_{x_discriminant}".replace("/", "_")


def _ymerged_2d_edges_path(
    mass: Mass,
    x_discriminant: str,
    y_discriminant: str,
) -> Path:
    mass = int(mass)
    safe_pair = _safe_pair_name(x_discriminant, y_discriminant)

    return (
        BDT_2D_BINNING_BASE
        / f"M{mass}"
        / "adaptive_discriminant_rebinning"
        / f"M{mass}"
        / "two_dimensional_discriminants"
        / BDT_2D_BINNING_SUBDIR
        / f"2D_{safe_pair}_yMergedMinWeightedBkg_edges_M{mass}_{BDT_ADAPTIVE_TAG}.json"
    )


# -------------------------------------------------------------------------
# Rectangular 2D binning from ColumnFlow variables
# -------------------------------------------------------------------------

def _edges_from_binning(binning) -> np.ndarray:
    """
    Convert a ColumnFlow/order variable binning into explicit bin edges.

    Supports:
      - regular binning: (n_bins, x_min, x_max)
      - variable binning: [edge0, edge1, edge2, ...]
    """
    b = list(binning)

    if (
        len(b) == 3
        and isinstance(b[0], (int, np.integer))
        and b[0] > 0
        and float(b[1]) < float(b[2])
    ):
        n_bins = int(b[0])
        x_min = float(b[1])
        x_max = float(b[2])

        return np.linspace(x_min, x_max, n_bins + 1, dtype=np.float64)

    edges = np.asarray(b, dtype=np.float64)

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(f"invalid variable binning: {binning}")

    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"bin edges are not strictly increasing: {edges}")

    return edges


def _get_1d_variable_edges(
    producer_inst: Producer,
    discriminant: str,
    mass: Mass,
) -> np.ndarray:
    """
    Read the binning of an existing 1D BDT discriminant variable.

    Example:
      discriminant = "D_sig", mass = 100
      -> reads cfg variable "bdt_D_sig_M100"
    """
    var_name = _column_name(discriminant, mass)

    try:
        var_inst = producer_inst.config_inst.get_variable(var_name)
    except Exception as exc:
        raise RuntimeError(
            f"cannot build 2D BDT variable because the 1D variable '{var_name}' "
            f"is not defined in the config"
        ) from exc

    return _edges_from_binning(var_inst.binning)


# -------------------------------------------------------------------------
# Irregular y-merged 2D binning from JSON
# -------------------------------------------------------------------------

def _extract_ymerged_2d_binning_from_json(data: dict) -> dict[str, object]:
    """
    Extract irregular 2D binning from the training JSON, e.g.

        {
          "x_edges": [...],
          "y_edges_by_x_bin": [
            {"x_bin": 1, "x_low": ..., "x_high": ..., "y_edges": [...]},
            ...
          ],
          "n_cells": 56
        }

    Returns explicit x edges, per-x-bin y edges, per-x-bin y-bin counts,
    cumulative x-bin offsets, and the total number of flattened 2D cells.
    """
    if not isinstance(data, dict):
        raise TypeError("y-merged 2D binning JSON payload is not a dictionary")

    if "x_edges" not in data:
        raise KeyError("missing key 'x_edges' in y-merged 2D binning JSON")
    if "y_edges_by_x_bin" not in data:
        raise KeyError("missing key 'y_edges_by_x_bin' in y-merged 2D binning JSON")

    x_edges = np.asarray([float(x) for x in data["x_edges"]], dtype=np.float64)

    if x_edges.ndim != 1 or len(x_edges) < 2:
        raise ValueError(f"invalid x_edges in y-merged 2D JSON: {x_edges}")
    if not np.all(np.diff(x_edges) > 0.0):
        raise ValueError(f"x_edges are not strictly increasing: {x_edges}")

    y_edges_by_x_bin = []
    n_y_bins_by_x_bin = []
    x_bin_offsets = []
    offset = 0

    entries = list(data["y_edges_by_x_bin"])
    if len(entries) != len(x_edges) - 1:
        raise ValueError(
            "inconsistent y-merged 2D JSON: "
            f"len(y_edges_by_x_bin)={len(entries)} but n_x_bins={len(x_edges) - 1}"
        )

    for ix, entry in enumerate(entries):
        if "y_edges" not in entry:
            raise KeyError(f"missing key 'y_edges' for x-bin entry {ix}")

        y_edges = np.asarray([float(y) for y in entry["y_edges"]], dtype=np.float64)

        if y_edges.ndim != 1 or len(y_edges) < 2:
            raise ValueError(f"invalid y_edges for x bin {ix}: {y_edges}")
        if not np.all(np.diff(y_edges) > 0.0):
            raise ValueError(f"y_edges are not strictly increasing for x bin {ix}: {y_edges}")

        n_y_bins = len(y_edges) - 1

        x_bin_offsets.append(offset)
        y_edges_by_x_bin.append(y_edges)
        n_y_bins_by_x_bin.append(n_y_bins)
        offset += n_y_bins

    n_cells_from_edges = int(offset)

    if "n_cells" in data:
        n_cells_declared = int(data["n_cells"])
        if n_cells_declared != n_cells_from_edges:
            raise ValueError(
                "inconsistent y-merged 2D JSON: "
                f"declared n_cells={n_cells_declared}, "
                f"computed n_cells={n_cells_from_edges}"
            )

    return {
        "x_edges": x_edges,
        "y_edges_by_x_bin": y_edges_by_x_bin,
        "n_y_bins_by_x_bin": np.asarray(n_y_bins_by_x_bin, dtype=np.int32),
        "x_bin_offsets": np.asarray(x_bin_offsets, dtype=np.int32),
        "n_cells": n_cells_from_edges,
    }


@functools.lru_cache(maxsize=None)
def _read_ymerged_2d_binning(
    mass: Mass,
    x_discriminant: str,
    y_discriminant: str,
) -> dict[str, object]:
    """
    Read and validate the irregular y-merged 2D binning JSON.
    """
    mass = int(mass)
    path = _ymerged_2d_edges_path(mass, x_discriminant, y_discriminant)

    if not path.is_file():
        raise RuntimeError(
            "Missing y-merged 2D BDT binning JSON for "
            f"{y_discriminant} vs {x_discriminant}, M={mass}:\n"
            f"  {path}\n"
            "This producer cannot safely fall back to rectangular flattening for "
            "this pair because the y binning depends on the x bin."
        )

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Could not read y-merged 2D BDT binning JSON: {path}") from exc

    # Validate against the intended pair when the JSON contains these fields.
    x_name = data.get("x_name")
    y_name = data.get("y_name")
    if x_name is not None and str(x_name) != str(x_discriminant):
        raise RuntimeError(
            f"Unexpected x_name in {path}: got {x_name!r}, expected {x_discriminant!r}"
        )
    if y_name is not None and str(y_name) != str(y_discriminant):
        raise RuntimeError(
            f"Unexpected y_name in {path}: got {y_name!r}, expected {y_discriminant!r}"
        )

    return _extract_ymerged_2d_binning_from_json(data)


# -------------------------------------------------------------------------
# Value access and flattening
# -------------------------------------------------------------------------

def _get_discriminant_values(
    events,
    discriminant: str,
    mass: Mass,
):
    """
    Return per-event values for a BDT discriminant.
    """
    if discriminant not in BDT_INPUTS:
        raise ValueError(f"unknown BDT discriminant: {discriminant}")

    column = _column_name(discriminant, mass)

    if column not in events.fields:
        raise RuntimeError(
            f"Missing required BDT input column '{column}'. "
            "Check that the current BDT-score producer was run before "
            "bdt_2d_variables."
        )

    return events[column]


def _ak_to_numpy_1d(values) -> np.ndarray:
    arr = ak.to_numpy(values)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _flatten_2d_rectangular(x, y, x_edges, y_edges):
    """
    Convert a rectangular 2D bin index (ix, iy) into a 1D flattened coordinate.

    Convention:

        flat_index = ix * n_y_bins + iy
        stored_value = flat_index + 0.5

    Invalid or out-of-range values are assigned -1.0.
    """
    x_np = _ak_to_numpy_1d(x)
    y_np = _ak_to_numpy_1d(y)

    if len(x_np) != len(y_np):
        raise RuntimeError(f"x/y length mismatch in rectangular 2D flattening: {len(x_np)} vs {len(y_np)}")

    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    ix = np.searchsorted(x_edges, x_np, side="right") - 1
    iy = np.searchsorted(y_edges, y_np, side="right") - 1

    # Include values exactly on the upper edge in the last bin.
    ix = np.where(x_np == x_edges[-1], nx - 1, ix)
    iy = np.where(y_np == y_edges[-1], ny - 1, iy)

    valid = (
        np.isfinite(x_np)
        & np.isfinite(y_np)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )

    flat_index = ix * ny + iy
    flat_value = np.where(valid, flat_index + 0.5, -1.0)

    return ak.Array(flat_value.astype(np.float32))


def _flatten_2d_ymerged(x, y, ymerged_binning: dict[str, object]):
    """
    Convert an irregular y-merged 2D bin index (ix, iy) into a 1D flattened
    coordinate.

    Convention:

        flat_index = x_bin_offset[ix] + iy
        stored_value = flat_index + 0.5

    Invalid or out-of-range values are assigned -1.0.
    """
    x_np = _ak_to_numpy_1d(x)
    y_np = _ak_to_numpy_1d(y)

    if len(x_np) != len(y_np):
        raise RuntimeError(f"x/y length mismatch in y-merged 2D flattening: {len(x_np)} vs {len(y_np)}")

    x_edges = np.asarray(ymerged_binning["x_edges"], dtype=np.float64)
    y_edges_by_x_bin = list(ymerged_binning["y_edges_by_x_bin"])
    x_bin_offsets = np.asarray(ymerged_binning["x_bin_offsets"], dtype=np.int32)

    nx = len(x_edges) - 1

    ix = np.searchsorted(x_edges, x_np, side="right") - 1
    ix = np.where(x_np == x_edges[-1], nx - 1, ix)

    valid_x = (
        np.isfinite(x_np)
        & np.isfinite(y_np)
        & (ix >= 0)
        & (ix < nx)
    )

    flat_value = np.full(len(x_np), -1.0, dtype=np.float32)

    for x_bin in range(nx):
        mask = valid_x & (ix == x_bin)
        if not np.any(mask):
            continue

        y_edges = np.asarray(y_edges_by_x_bin[x_bin], dtype=np.float64)
        ny = len(y_edges) - 1

        y_sel = y_np[mask]
        iy = np.searchsorted(y_edges, y_sel, side="right") - 1
        iy = np.where(y_sel == y_edges[-1], ny - 1, iy)

        valid_y = (iy >= 0) & (iy < ny)
        idx = np.where(mask)[0]

        flat_index = x_bin_offsets[x_bin] + iy
        flat_value[idx[valid_y]] = flat_index[valid_y].astype(np.float32) + 0.5

    return ak.Array(flat_value.astype(np.float32))


# -------------------------------------------------------------------------
# Producer
# -------------------------------------------------------------------------

@producer(
    uses=_bdt_2d_input_columns(
        MASS_POINTS,
        BDT_2D_PAIRS,
    ),
    produces=_bdt_2d_output_columns(
        MASS_POINTS,
        BDT_2D_PAIRS,
    ),
    mass_points=MASS_POINTS,
    bdt_2d_pairs=BDT_2D_PAIRS,
)
def bdt_2d_variables(
    self: Producer,
    events,
    **kwargs,):
    for mass in self.mass_points:
        for pair_name, x_disc, y_disc in self.bdt_2d_pairs:
            out_col = _pair_column_name(pair_name, mass)

            x_values = _get_discriminant_values(events, x_disc, mass)
            y_values = _get_discriminant_values(events, y_disc, mass)

            if (x_disc, y_disc) in BDT_YMERGED_2D_PAIRS:
                ymerged_binning = _read_ymerged_2d_binning(
                    int(mass),
                    x_disc,
                    y_disc,
                )

                flat_values = _flatten_2d_ymerged(
                    x_values,
                    y_values,
                    ymerged_binning,
                )

            else:
                x_edges = _get_1d_variable_edges(self, x_disc, mass)
                y_edges = _get_1d_variable_edges(self, y_disc, mass)

                flat_values = _flatten_2d_rectangular(
                    x_values,
                    y_values,
                    x_edges,
                    y_edges,
                )

            events = set_ak_column(
                events,
                out_col,
                flat_values,
            )

    return events

BDT_2D_CARD_BLOCK_PRODUCERS = {}


for block in get_bdt_mass_blocks():
    block = tuple(block)
    tag = get_bdt_mass_block_tag(block)

    cls_name = f"bdt_2d_card_{tag}"

    producer_cls = bdt_2d_variables.derive(
        cls_name,
        cls_dict={
            "mass_points": block,
            "bdt_2d_pairs": BDT_CARD_2D_PAIRS,
            "uses": _bdt_2d_input_columns(
                block,
                BDT_CARD_2D_PAIRS,
            ),
            "produces": _bdt_2d_output_columns(
                block,
                BDT_CARD_2D_PAIRS,
            ),
        },
    )

    globals()[cls_name] = producer_cls
    BDT_2D_CARD_BLOCK_PRODUCERS[block] = (
        producer_cls
    )