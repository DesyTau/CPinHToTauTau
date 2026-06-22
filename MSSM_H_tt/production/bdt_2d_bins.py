# MSSM_H_tt/production/bdt_2d_variables.py

from __future__ import annotations

from typing import Union

import numpy as np

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from MSSM_H_tt.config.mass_points import read_bdt_masses


ak = maybe_import("awkward")


MASS_POINTS = tuple(read_bdt_masses())

Mass = Union[int, str]


# Input columns already present in ProduceColumns:
#
#   bdt_D_sig_M{MASS}
#   bdt_D_ggphi_M{MASS}
#   bdt_D_bbphi_M{MASS}
#
BDT_INPUTS = ("D_sig", "D_ggphi", "D_bbphi")


# New 1D derived columns to produce:
#
#   bdt_Disc_ggphi_M{MASS}
#   bdt_Disc_bbphi_M{MASS}
#
BDT_DERIVED_1D = ("Disc_ggphi", "Disc_bbphi")


# 2D flattened variables to produce.
BDT_2D_PAIRS = (
    ("D_sig_vs_D_ggphi", "D_sig", "D_ggphi"),
    ("D_sig_vs_D_bbphi", "D_sig", "D_bbphi"),
    ("D_ggphi_vs_D_bbphi", "D_ggphi", "D_bbphi"),

    # New derived-disc 2D variables
    ("D_sig_vs_Disc_ggphi", "D_sig", "Disc_ggphi"),
    ("D_sig_vs_Disc_bbphi", "D_sig", "Disc_bbphi"),
)


# Binning convention for derived 1D discriminants.
#
# Disc_ggphi is assigned the binning of D_ggphi.
# Disc_bbphi is assigned the binning of D_bbphi.
#
DERIVED_DISC_BINNING_SOURCE = {
    "Disc_ggphi": "D_ggphi",
    "Disc_bbphi": "D_bbphi",
}


def _column_name(discriminant: str, mass: Mass) -> str:
    return f"bdt_{discriminant}_M{mass}"


def _pair_column_name(pair_name: str, mass: Mass) -> str:
    return f"bdt_{pair_name}_M{mass}"


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


def _get_variable_edges(
    producer_inst: Producer,
    discriminant: str,
    mass: Mass,
) -> np.ndarray:
    """
    Get bin edges for any discriminant used in 2D flattening.

    For saved discriminants:
      D_sig, D_ggphi, D_bbphi

    For derived discriminants:
      Disc_ggphi, Disc_bbphi

    The derived discriminants reuse the adaptive binning of D_ggphi and D_bbphi.
    """
    if discriminant in DERIVED_DISC_BINNING_SOURCE:
        source_discriminant = DERIVED_DISC_BINNING_SOURCE[discriminant]

        return _get_1d_variable_edges(
            producer_inst,
            source_discriminant,
            mass,
        )

    return _get_1d_variable_edges(
        producer_inst,
        discriminant,
        mass,
    )


def _safe_ratio(numerator, denominator):
    """
    Compute numerator / denominator safely.

    Values with denominator <= 0 are set to NaN.
    They will later become invalid bins and receive flat value -1.
    """
    safe_denominator = ak.where(
        denominator > 0,
        denominator,
        np.nan,
    )

    return numerator / safe_denominator


def _derived_disc_ggphi(events, mass: Mass):
    d_ggphi = events[_column_name("D_ggphi", mass)]
    d_bbphi = events[_column_name("D_bbphi", mass)]

    return _safe_ratio(
        d_ggphi,
        d_ggphi + d_bbphi,
    )


def _derived_disc_bbphi(events, mass: Mass):
    d_ggphi = events[_column_name("D_ggphi", mass)]
    d_bbphi = events[_column_name("D_bbphi", mass)]

    return _safe_ratio(
        d_bbphi,
        d_ggphi + d_bbphi,
    )


def _set_derived_1d_columns(events, mass: Mass):
    """
    Produce and save the 1D derived discriminants:

      bdt_Disc_ggphi_M{MASS}
      bdt_Disc_bbphi_M{MASS}
    """
    events = set_ak_column(
        events,
        _column_name("Disc_ggphi", mass),
        ak.values_astype(
            _derived_disc_ggphi(events, mass),
            np.float32,
        ),
    )

    events = set_ak_column(
        events,
        _column_name("Disc_bbphi", mass),
        ak.values_astype(
            _derived_disc_bbphi(events, mass),
            np.float32,
        ),
    )

    return events


def _get_discriminant_values(
    events,
    discriminant: str,
    mass: Mass,
):
    """
    Return per-event values for a discriminant.

    Saved discriminants are read directly from events.
    Derived discriminants are read from the columns produced earlier in this producer.
    """
    if discriminant in BDT_INPUTS or discriminant in BDT_DERIVED_1D:
        return events[_column_name(discriminant, mass)]

    raise ValueError(f"unknown BDT discriminant: {discriminant}")


def _flatten_2d(x, y, x_edges, y_edges):
    """
    Convert a 2D bin index (ix, iy) into a 1D flattened bin coordinate.

    Convention:

        flat_index = ix * n_y_bins + iy

    The returned value is flat_index + 0.5, so the ColumnFlow variable should use:

        binning = (n_x_bins * n_y_bins, 0, n_x_bins * n_y_bins)

    Invalid or out-of-range values are assigned -1.0.
    """
    x_np = ak.to_numpy(x)
    y_np = ak.to_numpy(y)

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


@producer(
    uses={
        _column_name(discriminant, mass)
        for mass in MASS_POINTS
        for discriminant in BDT_INPUTS
    },
    produces=(
        {
            _column_name(discriminant, mass)
            for mass in MASS_POINTS
            for discriminant in BDT_DERIVED_1D
        }
        |
        {
            _pair_column_name(pair_name, mass)
            for mass in MASS_POINTS
            for pair_name, _, _ in BDT_2D_PAIRS
        }
    ),
)
def bdt_2d_variables(self: Producer, events, **kwargs):
    for mass in MASS_POINTS:
        # First produce the missing 1D derived variables.
        events = _set_derived_1d_columns(events, mass)

        # Then produce the flattened 2D variables.
        for pair_name, x_disc, y_disc in BDT_2D_PAIRS:
            out_col = _pair_column_name(pair_name, mass)

            x_edges = _get_variable_edges(self, x_disc, mass)
            y_edges = _get_variable_edges(self, y_disc, mass)

            x_values = _get_discriminant_values(events, x_disc, mass)
            y_values = _get_discriminant_values(events, y_disc, mass)

            events = set_ak_column(
                events,
                out_col,
                _flatten_2d(
                    x_values,
                    y_values,
                    x_edges,
                    y_edges,
                ),
            )

    return events