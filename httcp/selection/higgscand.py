# coding: utf-8

"""
Prepare h-Candidate from SelectionResult: selected lepton indices & channel_id [trigger matched] 
"""

from typing import Optional
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@selector(
    uses={
        "channel_id",
    },
    produces={
        "hcand",
    },
    exposed=False,
)
def higgscand(
        self: Selector,
        events: ak.Array,
        hcand_pair: ak.Array,
        **kwargs
) -> tuple[ak.Array, SelectionResult]:
    sel_hcand = ak.sum(ak.num(hcand_pair, axis=-1), axis=-1) > 0
    #print(ak.to_list(sel_hcand))
    hcand_col = ak.firsts(hcand_pair, axis=1)
    #print(ak.to_list(hcand_col.pt))
    events = set_ak_column(events, "hcand", hcand_col)
    
    return events, SelectionResult(
        steps={
            "Atleast_one_higgs_cand": sel_hcand,
        },
    )
