# coding: utf-8
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Iterable, List
from columnflow.util import maybe_import
import re

_SIGNAL_DATASET_MASS_RE = re.compile(
    r"^(?:ggphi|bbphi)_phitt_([0-9]+)$"
)

def get_mssm_signal_mass(
    dataset_or_name,
) -> int | None:
    """
    Return the MSSM signal mass encoded in a dataset or process name.

    Examples:

        ggphi_phitt_500  -> 500
        bbphi_phitt_1200 -> 1200
        DY...            -> None
        data...          -> None

    The input can either be an order.Dataset instance or a string.
    """

    if dataset_or_name is None:
        return None

    name = (
        dataset_or_name.name
        if hasattr(
            dataset_or_name,
            "name",
        )
        else str(
            dataset_or_name
        )
    )

    match = _SIGNAL_DATASET_MASS_RE.match(
        name
    )

    if not match:
        return None

    return int(
        match.group(1)
    )

def get_bdt_masses_for_dataset(
    dataset_inst,
    masses=None,
) -> tuple[int, ...]:
    """
    Return the BDT masses that should be evaluated for a dataset.

    Backgrounds and data:
        all requested masses

    MSSM signals:
        only the mass corresponding to the signal dataset

    Examples:

        bbphi_phitt_100 -> (100,)
        ggphi_phitt_100 -> (100,)
        DY...           -> all masses
        data...         -> all masses
    """

    if masses is None:
        masses = read_bdt_masses()

    masses = tuple(
        int(mass)
        for mass in masses
    )

    signal_mass = get_mssm_signal_mass(
        dataset_inst
    )

    # Background or data.
    if signal_mass is None:
        return masses

    # Signal.
    if signal_mass not in masses:
        raise ValueError(
            f"signal dataset "
            f"'{dataset_inst.name}' "
            f"has mass {signal_mass}, but "
            f"this mass is not contained "
            f"in the requested BDT masses "
            f"{masses}"
        )

    return (
        signal_mass,
    )
  
@lru_cache(maxsize=1)
def read_bdt_masses(path: str | Path | None = None) -> List[int]:
  """
  Load MSSM mass points from bdt_masses.yaml.
  Accepts either:
    - a dict with key 'masses'
    - a plain top-level YAML list
  Caches the result.
  """
  yaml = maybe_import("yaml")
  if yaml is None:
    raise RuntimeError("PyYAML is required to load mass points. Please `pip install pyyaml`.")

  p = Path(path) if path else Path(__file__).with_name("bdt_masses.yaml")
  if not p.exists():
    raise FileNotFoundError(f"Mass list file not found: {p}")

  data = yaml.safe_load(p.read_text())

  if isinstance(data, dict) and "masses" in data:
    masses = data["masses"]
  elif isinstance(data, list):
    masses = data
  else:
    raise ValueError("bdt_masses.yaml must be either a list or a dict with key 'masses'.")

  try:
    out = [int(m) for m in masses]
  except Exception as exc:
    raise ValueError("Failed to parse masses as integers from bdt_masses.yaml") from exc

  # optional: keep author order, but guard against accidental duplicates
  # (no sorting to preserve intended registration order)
  seen = set()
  uniq = []
  for m in out:
    if m not in seen:
      uniq.append(m)
      seen.add(m)
  return uniq

BDT_MASS_BLOCK_SIZE = 0


def get_bdt_mass_blocks(
    block_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if block_size is None:
        block_size = BDT_MASS_BLOCK_SIZE

    masses = tuple(read_bdt_masses())

    # block size <= 0 means: process all configured masses together
    if block_size <= 0 or block_size >= len(masses):
        return (masses,)

    return tuple(
        tuple(masses[i:i + block_size])
        for i in range(0, len(masses), block_size)
    )


def get_bdt_mass_block(
    mass: int,
    block_size: int | None = None,
) -> tuple[int, ...]:
    mass = int(mass)

    for block in get_bdt_mass_blocks(block_size):
        if mass in block:
            return block

    raise ValueError(
        f"Mass {mass} not found in configured BDT masses "
        f"{read_bdt_masses()}"
    )
BDT_CARD_HIST_VARIABLES = (
    "D_sig_vs_Disc_ggphi",
    "D_sig_vs_Disc_bbphi",
    "D_DY",
    "D_TT",
)

_BDT_CARD_HIST_VARIABLE_RE = re.compile(
    r"^bdt_"
    r"(D_sig_vs_Disc_ggphi|"
    r"D_sig_vs_Disc_bbphi|"
    r"D_DY|D_TT)"
    r"_M([0-9]+)$"
)

def get_mssm_signal_mass(
    dataset_or_name,
) -> int | None:
    """
    Return the MSSM signal mass encoded in a dataset/process name.

    Examples:

        ggphi_phitt_500 -> 500
        bbphi_phitt_500 -> 500
        DY...           -> None
        data...         -> None
    """

    if dataset_or_name is None:
        return None

    name = (
        dataset_or_name.name
        if hasattr(dataset_or_name, "name")
        else str(dataset_or_name)
    )

    match = _SIGNAL_DATASET_MASS_RE.match(
        name
    )

    if not match:
        return None

    return int(
        match.group(1)
    )
    
def expand_bdt_histogram_variables(
    variables,
    dataset=None,
) -> tuple[str, ...]:
    """
    Preserve requested BDT histogram variables exactly as requested.

    Rules
    -----
    Background and data datasets:
        Keep the requested variable unchanged.

    Signal datasets:
        Keep the requested variable unchanged, but require the mass encoded
        in the variable to match the mass encoded in the signal dataset.

    Examples
    --------

    Background:

        dataset:
            DYto2L_M_50_amcatnloFXFX

        requested:
            bdt_D_sig_vs_Disc_ggphi_M100

        result:
            bdt_D_sig_vs_Disc_ggphi_M100


    Matching signal:

        dataset:
            ggphi_phitt_100

        requested:
            bdt_D_sig_vs_Disc_ggphi_M100

        result:
            bdt_D_sig_vs_Disc_ggphi_M100


    Mismatching signal:

        dataset:
            ggphi_phitt_500

        requested:
            bdt_D_sig_vs_Disc_ggphi_M100

        result:
            ValueError

    The histogram-variable expander must never silently change the requested
    mass or add unrelated BDT discriminants.
    """

    expanded = []
    seen = set()

    # Signal mass, or None for backgrounds/data/no dataset.
    signal_mass = get_mssm_signal_mass(
        dataset
    )

    # Useful for error messages.
    if dataset is None:
        dataset_name = None
    elif hasattr(dataset, "name"):
        dataset_name = dataset.name
    else:
        dataset_name = str(dataset)

    for variable in variables:

        variable = str(variable)

        match = _BDT_CARD_HIST_VARIABLE_RE.match(
            variable
        )

        # -------------------------------------------------------------
        # BDT datacard variable
        # -------------------------------------------------------------
        if match:

            requested_mass = int(
                match.group(2)
            )

            # For MSSM signal datasets, explicitly require the signal mass
            # and requested BDT mass to agree.
            if (
                signal_mass is not None
                and signal_mass != requested_mass
            ):
                raise ValueError(
                    "Inconsistent MSSM signal dataset and BDT histogram "
                    "variable:\n"
                    f"  dataset        : {dataset_name}\n"
                    f"  signal mass    : {signal_mass}\n"
                    f"  variable       : {variable}\n"
                    f"  requested mass : {requested_mass}\n"
                    "\n"
                    "The histogram variable mass must match the signal "
                    "dataset mass."
                )

        # -------------------------------------------------------------
        # Preserve variable exactly as requested.
        # -------------------------------------------------------------
        if variable not in seen:
            expanded.append(
                variable
            )
            seen.add(
                variable
            )

    return tuple(
        expanded
    )

def get_bdt_mass_block_tag(
    masses,
) -> str:
    masses = tuple(int(m) for m in masses)

    if masses == tuple(read_bdt_masses()):
        return "all"

    return "M" + "_".join(str(m) for m in masses)


def get_bdt_card_producer_name(
    mass: int,
) -> str:
    block = get_bdt_mass_block(mass)

    return f"bdt_card_{get_bdt_mass_block_tag(block)}"
