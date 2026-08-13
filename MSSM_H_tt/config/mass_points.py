# coding: utf-8
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Iterable, List
from columnflow.util import maybe_import

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

BDT_MASS_BLOCK_SIZE = 6


def get_bdt_mass_blocks(
    block_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if block_size is None:
        block_size = BDT_MASS_BLOCK_SIZE

    if block_size <= 0:
        raise ValueError(
            f"BDT mass block size must be positive, got {block_size}"
        )

    masses = tuple(read_bdt_masses())

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


def get_bdt_mass_block_tag(
    masses,
) -> str:
    return "M" + "_".join(str(int(m)) for m in masses)


def get_bdt_card_producer_name(
    mass: int,
) -> str:
    block = get_bdt_mass_block(mass)

    return f"bdt_card_{get_bdt_mass_block_tag(block)}"
