"""
Produce channel_id column. This function is called in the main selector
"""

from columnflow.production import Producer, producer
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import, DotDict
from MSSM_H_tt.util import get_lep_p4, find_fields_with_nan, get_p2
from columnflow.columnar_util import EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")
import functools

@producer(
    uses={
        "RecoilCorrMET.{pt,phi}",
        "hcand_*",
    },
    produces={"pt_H"},
    exposed=False,
)
def pt_H(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Produce pt_H using recoil-corrected MET.

    This allows both:
      - unclustered MET shifts, propagated through RecoilCorrMET
      - recoil response/resolution shifts, via RecoilCorrMET aliases
    """

    electron = events.hcand_emu.lep0
    muon = events.hcand_emu.lep1

    somma = (
        get_p2(electron)
        + get_p2(muon)
        + get_p2(events.RecoilCorrMET)
    )

    pt_H_value = somma.rho

    events = set_ak_column(events, "pt_H", pt_H_value)
    return events


@pt_H.init
def pt_H_init(self: Producer) -> None:
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("met", "met_recoil"))
    }