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
    Expand BDT datacard histogram variables.

    For backgrounds and data:
        requesting one BDT datacard variable expands to the complete
        configured BDT mass block.

    For MSSM signals:
        requesting any BDT datacard variable expands only to the four
        datacard variables corresponding to the mass of the signal
        dataset.

    Ordinary variables are left unchanged.
    """

    expanded = []
    seen = set()

    def add(
        variable,
    ):
        if variable not in seen:
            expanded.append(
                variable
            )
            seen.add(
                variable
            )

    # Determine whether this is an MSSM signal dataset.
    signal_mass = get_mssm_signal_mass(
        dataset
    )

    for variable in variables:

        match = _BDT_CARD_HIST_VARIABLE_RE.match(
            variable
        )

        # -------------------------------------------------------------
        # Ordinary variable
        #
        # Examples:
        #   emu_mt_tot
        #   emu_mvis
        #   D_zeta
        #
        # Nothing special to do.
        # -------------------------------------------------------------

        if not match:
            add(
                variable
            )
            continue

        requested_mass = int(
            match.group(2)
        )

        # -------------------------------------------------------------
        # MSSM signal
        #
        # Ignore the mass encoded in the seed histogram variable and
        # use the mass corresponding to the signal dataset itself.
        #
        # Example:
        #
        #   dataset:
        #       ggphi_phitt_500
        #
        #   requested:
        #       bdt_D_DY_M60
        #
        #   resulting mass block:
        #       (500,)
        #
        # This allows MergeShiftedHistogramsWrapper to use the same
        # symbolic seed variable for every signal dataset.
        # -------------------------------------------------------------

        if signal_mass is not None:

            block = (
                signal_mass,
            )

        # -------------------------------------------------------------
        # Background or data
        #
        # Preserve the existing mass-block behavior.
        #
        # With BDT_MASS_BLOCK_SIZE = 0:
        #
        #   bdt_D_DY_M60
        #
        # expands to all configured masses.
        # -------------------------------------------------------------

        else:

            block = get_bdt_mass_block(
                requested_mass
            )

        # -------------------------------------------------------------
        # For every active mass, request all four final BDT histogram
        # variables.
        # -------------------------------------------------------------

        for block_mass in block:

            for discriminant in (
                BDT_CARD_HIST_VARIABLES
            ):

                add(
                    f"bdt_"
                    f"{discriminant}"
                    f"_M{block_mass}"
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
