# coding: utf-8

"""
Main categories file for the Higgs MSSM analysis
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

from types import FunctionType
from copy import copy

ak = maybe_import("awkward")
np = maybe_import("numpy")

#
# categorizer functions used by categories definitions
#

def copy_function(fn, name):
    return FunctionType(
    copy(fn.__code__),
    copy(fn.__globals__),
    name=name,
    argdefs=copy(fn.__defaults__),
    closure=copy(fn.__closure__)
)
    
@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1

#Four general categories: etau, mutau, emu and tautau
@categorizer(uses={'hcand_emu.*'})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.num(events.hcand_emu.lep0.pt > 0, axis =1) > 0
    return events, mask 

@categorizer(uses={'event', 'hcand_emu.lep1.pfRelIso04_all'})
def lep_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channel = self.config_inst.channels.names()[0] #We are processing a single channel at once
    if channel == 'emu': 
        isolation = events.hcand_emu.lep1.pfRelIso04_all < 0.2
    else:
        raise NotImplementedError(
                f'Can not find an isolation criteria for {channel} channel!')
    mask = ak.fill_none(ak.firsts(isolation, axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def lep_inv_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channel = self.config_inst.channels.names()[0] #We are processing a single channel at once
    if channel == 'emu':
        isolation = events.hcand_emu.lep1.pfRelIso04_all >= 0.2
        upper_lim = events.hcand_emu.lep1.pfRelIso04_all < 0.5
    else:
        raise NotImplementedError(
                f'Can not find an isolation criteria for {channel} channel!')
    mask = ak.fill_none(ak.firsts(isolation, axis=1),False)
    mask = mask & ak.fill_none(ak.firsts(upper_lim, axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def os_charge(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].rel_charge < 0), axis=1),False)
    return events, mask

@categorizer(uses={'event', 'hcand_*'})
def ss_charge(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    channels = self.config_inst.channels.names()
    mask = ak.zeros_like(events.event, dtype=np.bool_)
    for ch_str in channels:
        mask = mask | ak.fill_none(ak.firsts((events[f'hcand_{ch_str}'].rel_charge > 0), axis=1),False)
    return events, mask

@categorizer(uses={"N_b_jets"})
def Zero_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets == 0 
    return events, mask

@categorizer(uses={"N_b_jets"})
def One_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets == 1 
    return events, mask

@categorizer(uses={"N_b_jets"})
def At_least_1_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets >= 1
    return events, mask

@categorizer(uses={"N_b_jets"})
def At_least_2_b_jets(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.N_b_jets >= 2 
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_low(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -35) & (events.D_zeta < -10)
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_mid(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -10) 
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut_high(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= 30)
    return events, mask

@categorizer(uses={'D_zeta'})
def D_zeta_cut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.D_zeta >= -80)
    return events, mask

# Higgs BDT score categories ---------------------------------------------------
#
# The score producer writes the four raw class probabilities
#
#   bdt_raw_score_ggphi_M{mass}
#   bdt_raw_score_bbphi_M{mass}
#   bdt_raw_score_dy_M{mass}
#   bdt_raw_score_tt_M{mass}
#
# and the four-class argmax category
#
#   bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)
#
# The new 10-feature training also uses a three-region merged convention in
# several post-processing steps:
#
#   signal region -> max(P_ggphi + P_bbphi, P_DY, P_TT) = P_ggphi + P_bbphi
#   DY region     -> max(P_ggphi + P_bbphi, P_DY, P_TT) = P_DY
#   TT region     -> max(P_ggphi + P_bbphi, P_DY, P_TT) = P_TT
#
# Both conventions are exposed below.  The default signal/DY/TT names use the
# merged convention, while explicit four-class names are kept for diagnostics.


def _bdt_raw_score_fields(mass: int) -> dict[str, str]:
    return {
        "ggphi": f"bdt_raw_score_ggphi_M{mass}",
        "bbphi": f"bdt_raw_score_bbphi_M{mass}",
        "dy": f"bdt_raw_score_dy_M{mass}",
        "tt": f"bdt_raw_score_tt_M{mass}",
    }


def _check_bdt_fields(events: ak.Array, required_fields: list[str], context: str) -> None:
    missing_fields = [
        field
        for field in required_fields
        if field not in events.fields
    ]

    if missing_fields:
        raise RuntimeError(
            f"Missing BDT fields for {context}: {missing_fields}. "
            "Check that the BDT-score producer was run before categorization."
        )


def _bdt_cat_fourclass_mass(
    self: Categorizer,
    events: ak.Array,
    cat_id: int,
    mass: int,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Per-mass four-class BDT region selection.

    Definition:

        bdt_cat_M{mass} = argmax(P_ggphi, P_bbphi, P_DY, P_TT)

    Convention:

        0 -> ggphi
        1 -> bbphi
        2 -> DY
        3 -> TT
    """
    field = f"bdt_cat_M{mass}"

    _check_bdt_fields(
        events,
        [field],
        context=f"four-class BDT category M={mass}",
    )

    mask = events[field] == int(cat_id)

    return events, ak.fill_none(mask, False)


def _bdt_merged_scores_mass(
    events: ak.Array,
    mass: int,
) -> tuple[ak.Array, ak.Array, ak.Array]:
    """
    Return the three merged-region scores:

        P_signal = P_ggphi + P_bbphi
        P_DY
        P_TT
    """
    fields = _bdt_raw_score_fields(mass)

    _check_bdt_fields(
        events,
        list(fields.values()),
        context=f"merged BDT category M={mass}",
    )

    p_signal = events[fields["ggphi"]] + events[fields["bbphi"]]
    p_dy = events[fields["dy"]]
    p_tt = events[fields["tt"]]

    return p_signal, p_dy, p_tt


def _bdt_cat_merged_mass(
    self: Categorizer,
    events: ak.Array,
    merged_cat_id: int,
    mass: int,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Per-mass three-region BDT category selection.

    Definition:

        merged_cat = argmax(P_ggphi + P_bbphi, P_DY, P_TT)

    Convention:

        0 -> signal = ggphi + bbphi
        1 -> DY
        2 -> TT
    """
    p_signal, p_dy, p_tt = _bdt_merged_scores_mass(events, mass)

    if int(merged_cat_id) == 0:
        mask = (p_signal >= p_dy) & (p_signal >= p_tt)
    elif int(merged_cat_id) == 1:
        mask = (p_dy > p_signal) & (p_dy >= p_tt)
    elif int(merged_cat_id) == 2:
        mask = (p_tt > p_signal) & (p_tt > p_dy)
    else:
        raise ValueError(
            f"Unknown merged BDT category id {merged_cat_id}. "
            "Expected 0=signal, 1=DY, 2=TT."
        )

    return events, ak.fill_none(mask, False)


from MSSM_H_tt.config.mass_points import read_bdt_masses

MASS_POINTS = read_bdt_masses()


BDT_REGION_SPECS = {
    # Main three-region convention used by the new 10-feature training
    # post-processing and by the merged-region plots.
    "signal": {
        "kind": "merged",
        "cat_id": 0,
        "description": "Merged BDT region where P_ggphi + P_bbphi is maximal",
    },
    "ggphi_and_bbphi": {
        "kind": "merged",
        "cat_id": 0,
        "description": "Alias of signal: merged BDT region where P_ggphi + P_bbphi is maximal",
    },
    "dy": {
        "kind": "merged",
        "cat_id": 1,
        "description": "Merged BDT region where P_DY is maximal against P_ggphi + P_bbphi and P_TT",
    },
    "tt": {
        "kind": "merged",
        "cat_id": 2,
        "description": "Merged BDT region where P_TT is maximal against P_ggphi + P_bbphi and P_DY",
    },

    # Explicit four-class regions kept for diagnostics and backwards-compatible
    # control plots.  These use bdt_cat_M{mass} directly.
    "ggphi": {
        "kind": "fourclass",
        "cat_id": 0,
        "description": "Four-class BDT region where P_ggphi is maximal",
    },
    "bbphi": {
        "kind": "fourclass",
        "cat_id": 1,
        "description": "Four-class BDT region where P_bbphi is maximal",
    },
    "dy_fourclass": {
        "kind": "fourclass",
        "cat_id": 2,
        "description": "Four-class BDT region where P_DY is maximal",
    },
    "tt_fourclass": {
        "kind": "fourclass",
        "cat_id": 3,
        "description": "Four-class BDT region where P_TT is maximal",
    },
}


BDT_REGION_DESCRIPTIONS = {
    region_name: spec["description"]
    for region_name, spec in BDT_REGION_SPECS.items()
}


for mass in MASS_POINTS:
    for region_name, spec in BDT_REGION_SPECS.items():
        if spec["kind"] == "fourclass":
            cat_id = spec["cat_id"]

            # Capture loop variables via defaults to avoid late binding.
            tmp_func = (
                lambda self, events, _cat_id=cat_id, _mass=mass, **kwargs:
                    _bdt_cat_fourclass_mass(
                        self,
                        events,
                        cat_id=_cat_id,
                        mass=_mass,
                        **kwargs,
                    )
            )

            uses = {
                f"bdt_cat_M{mass}",
            }

        elif spec["kind"] == "merged":
            cat_id = spec["cat_id"]

            # Capture loop variables via defaults to avoid late binding.
            tmp_func = (
                lambda self, events, _cat_id=cat_id, _mass=mass, **kwargs:
                    _bdt_cat_merged_mass(
                        self,
                        events,
                        merged_cat_id=_cat_id,
                        mass=_mass,
                        **kwargs,
                    )
            )

            fields = _bdt_raw_score_fields(mass)
            uses = set(fields.values())

        else:
            raise ValueError(
                f"Unknown BDT region kind '{spec['kind']}' "
                f"for region '{region_name}'."
            )

        globals()[f"bdt_cat_{region_name}_M{mass}"] = categorizer(
            copy_function(tmp_func, f"bdt_cat_{region_name}_M{mass}"),
            uses=uses,
        )