# coding: utf-8

"""
Exemplary selection methods.
"""

from typing import Optional
from operator import and_
from functools import reduce
from collections import defaultdict, OrderedDict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters

from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.util import attach_coffea_behavior

from columnflow.util import maybe_import
from columnflow.columnar_util import optional_column as optional
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from httcp.production.main import cutflow_features

from httcp.selection.physics_objects import *
from httcp.selection.trigger import trigger_selection
from httcp.selection.lepton_pair_etau import etau_selection
from httcp.selection.lepton_pair_mutau import mutau_selection
from httcp.selection.lepton_pair_tautau import tautau_selection
from httcp.selection.event_category import get_categories
from httcp.selection.match_trigobj import match_trigobj
from httcp.selection.lepton_veto import *
from httcp.selection.higgscand import higgscand, higgscandprod

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")


def manual_transform(col: ak.Array, decayMode_: Optional[int]=None):
    decayMode = None
    pt  = ak.values_astype(col.pt, "float64")
    eta = ak.values_astype(col.eta, "float64")
    phi = ak.values_astype(col.phi, "float64")
    mass = ak.values_astype(col.mass, "float64")
    charge = ak.values_astype(col.charge, "float64")
    if "decayMode" in col.fields:
        decayMode = col.decayMode
    else:
        if decayMode_ is not None:
            decayMode = decayMode_ * ak.ones_like(col.pt, dtype="float64")

    common = ak.zip(
        {
            "pt": col.pt,
            "eta": col.eta,
            "phi": col.phi,
            "mass": col.mass,
            "charge": col.charge,
            #"decayMode": col.decayMode if "decayMode" in col.fields else -99 * ak.ones_like(col.pt),
            "decayMode": decayMode,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=coffea.nanoevents.methods.vector.behavior,
    )
    return common


@selector(uses={"process_id", optional("mc_weight")})
def custom_increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # get event masks
    event_mask = results.event

    # get a list of unique process ids present in the chunk
    unique_process_ids = np.unique(events.process_id)
    # increment plain counts
    n_evt_per_file = self.dataset_inst.n_events/self.dataset_inst.n_files
    #from IPython import embed
    #embed()
    stats["num_events"] = n_evt_per_file
    stats["num_events_selected"] += ak.sum(event_mask, axis=0)
    if self.dataset_inst.is_mc:
        stats[f"sum_mc_weight"] = n_evt_per_file
        stats.setdefault(f"sum_mc_weight_per_process", defaultdict(float))
        for p in unique_process_ids:
            stats[f"sum_mc_weight_per_process"][int(p)] = n_evt_per_file
        
    # create a map of entry names to (weight, mask) pairs that will be written to stats
    weight_map = OrderedDict()
    if self.dataset_inst.is_mc:
        # mc weight for selected events
        weight_map["mc_weight_selected"] = (events.mc_weight, event_mask)

    # get and store the sum of weights in the stats dictionary
    for name, (weights, mask) in weight_map.items():
        joinable_mask = True if mask is Ellipsis else mask

        # sum of different weights in weight_map for all processes
        stats[f"sum_{name}"] += ak.sum(weights[mask])
        # sums per process id
        stats.setdefault(f"sum_{name}_per_process", defaultdict(float))
        for p in unique_process_ids:
            stats[f"sum_{name}_per_process"][int(p)] += ak.sum(
                weights[(events.process_id == p) & joinable_mask],
            )

    return events, results


# exposed selectors
# (those that can be invoked from the command line)
@selector(
    uses={
        "event",
        # selectors / producers called within _this_ selector
        attach_coffea_behavior,
        json_filter, 
        met_filters, 
        mc_weight, 
        cutflow_features, 
        process_ids,
        trigger_selection, 
        muon_selection, 
        electron_selection, 
        tau_selection, 
        jet_selection,
        etau_selection, 
        mutau_selection, 
        tautau_selection, 
        get_categories,
        extra_lepton_veto, 
        double_lepton_veto, 
        match_trigobj,
        increment_stats, 
        custom_increment_stats,
        #hcand_features,
        higgscand,
        higgscandprod, 
        #push_hcand,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        #"Muon.*",
        #"Electron.*",
        #"Tau.*",
        #"PV.*",
        #"MET.*",
        mc_weight, 
        trigger_selection, 
        muon_selection, 
        electron_selection, 
        tau_selection, 
        jet_selection,
        etau_selection, 
        mutau_selection, 
        tautau_selection, 
        get_categories, 
        cutflow_features, 
        process_ids,
        extra_lepton_veto, 
        double_lepton_veto, 
        match_trigobj, 
        #hcand_features, 
        #higgscand,
        higgscandprod,
        #push_hcand,
        #"hcand.*",
    },
    exposed=True,
)
def main(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    #events = set_ak_column(events, "Electron.decayMode", -1)
    #events = set_ak_column(events, "Muon.decayMode", -2)
    
    # ensure coffea behaviors are loaded
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # trigger selection
    events, trigger_results = self[trigger_selection](events, **kwargs)
    results += trigger_results

    # met filter selection
    events, met_filter_results = self[met_filters](events, **kwargs)
    results += met_filter_results
    
    # jet selection
    events, bjet_veto_result = self[jet_selection](events, 
                                                   call_force=True, 
                                                   **kwargs)
    results += bjet_veto_result

    # muon selection
    # e.g. mu_idx: [ [0,1], [], [1], [0], [] ] 
    events, muon_results, good_muon_indices, veto_muon_indices, dlveto_muon_indices = self[muon_selection](events,
                                                                                                           call_force=True, 
                                                                                                           **kwargs)
    results += muon_results

    # electron selection
    # e.g. ele_idx: [ [], [0,1], [], [], [1,2] ] 
    events, ele_results, good_ele_indices, veto_ele_indices, dlveto_ele_indices = self[electron_selection](events,
                                                                                                           call_force=True, 
                                                                                                           **kwargs)
    results += ele_results

    # tau selection
    # e.g. tau_idx: [ [1], [0,1], [1,2], [], [0,1] ] 
    events, tau_results, good_tau_indices = self[tau_selection](events,
                                                                good_ele_indices,
                                                                good_muon_indices,
                                                                call_force=True, 
                                                                **kwargs)
    results += tau_results

    _lepton_indices = ak.concatenate([good_muon_indices, good_ele_indices, good_tau_indices], axis=1)
    results.steps["At least two leptons of any sign"] = ak.num(_lepton_indices, axis=1) >= 2

    # trigger obj matching
    # INFO: for now, it is switched off
    events, good_ele_indices, good_muon_indices, good_tau_indices = self[match_trigobj](events,
                                                                                        trigger_results,
                                                                                        good_ele_indices,
                                                                                        good_muon_indices,
                                                                                        good_tau_indices,
                                                                                        False)

    # double lepton veto
    events, extra_double_lepton_veto_results = self[double_lepton_veto](events,
                                                                        dlveto_ele_indices,
                                                                        dlveto_muon_indices)
    results += extra_double_lepton_veto_results
    
    # e-tau pair i.e. hcand selection
    # e.g. [ [], [e1, tau1], [], [], [e1, tau2] ]
    etau_results, etau_indices_pair = self[etau_selection](events,
                                                           good_ele_indices,
                                                           good_tau_indices,
                                                           call_force=True,
                                                           **kwargs)
    results += etau_results

    

    #etau_pair         = ak.concatenate([manual_transform(events.Electron[etau_indices_pair[:,0:1]], -1),
    #                                    manual_transform(events.Tau[etau_indices_pair[:,1:2]])],
    #                                   axis=1)
    etau_pair         = ak.concatenate([events.Electron[etau_indices_pair[:,0:1]],
                                        events.Tau[etau_indices_pair[:,1:2]]],
                                       axis=1)

    #from IPython import embed; embed()
    #1/0

    # mu-tau pair i.e. hcand selection
    # e.g. [ [mu1, tau1], [], [mu1, tau2], [], [] ]
    mutau_results, mutau_indices_pair = self[mutau_selection](events,
                                                              good_muon_indices,
                                                              good_tau_indices,
                                                              call_force=True,
                                                              **kwargs)
    results += mutau_results

    #mutau_pair = ak.concatenate([manual_transform(events.Muon[mutau_indices_pair[:,0:1]], -2), 
    #                             manual_transform(events.Tau[mutau_indices_pair[:,1:2]])],
    #                            axis=1)
    mutau_pair = ak.concatenate([events.Muon[mutau_indices_pair[:,0:1]], 
                                 events.Tau[mutau_indices_pair[:,1:2]]],
                                axis=1)

    # tau-tau pair i.e. hcand selection
    # e.g. [ [], [tau1, tau2], [], [], [] ]
    tautau_results, tautau_indices_pair = self[tautau_selection](events,
                                                                 good_tau_indices,
                                                                 call_force=True,
                                                                 **kwargs)
    results += tautau_results

    #tautau_pair = ak.concatenate([manual_transform(events.Tau[tautau_indices_pair[:,0:1]]), 
    #                              manual_transform(events.Tau[tautau_indices_pair[:,1:2]])], 
    #                             axis=1)
    tautau_pair = ak.concatenate([events.Tau[tautau_indices_pair[:,0:1]], 
                                  events.Tau[tautau_indices_pair[:,1:2]]], 
                                 axis=1)

    # channel selection
    # channel_id is now in columns
    events, channel_results = self[get_categories](events,
                                                   trigger_results,
                                                   etau_indices_pair,
                                                   mutau_indices_pair,
                                                   tautau_indices_pair)
    results += channel_results

    # make sure events have at least one lepton pair
    # hcand pair: [ [[mu1,tau1]], [[e1,tau1],[tau1,tau2]], [[mu1,tau2]], [], [[e1,tau2]] ]
    hcand_pairs = ak.concatenate([etau_pair[:,None], mutau_pair[:,None], tautau_pair[:,None]], axis=1)

    #from IPython import embed; embed()

    # extra lepton veto
    # it is only applied on the events with one higgs candidate only
    events, extra_lepton_veto_results = self[extra_lepton_veto](events,
                                                                veto_ele_indices,
                                                                veto_muon_indices,
                                                                hcand_pairs)
    results += extra_lepton_veto_results


    # hcand results
    events, hcand_array, hcand_results = self[higgscand](events, hcand_pairs)
    #events = self[higgscand](events, hcand_pairs)
    #hcand_results = SelectionResult(steps={"atleast_one_higgs_cand_per_event": ak.num(ak.firsts(events.hcand.pt, axis=1), axis=1) == 2})
    #events = self[push_hcand](events, hcand_array, **kwargs)    
    results += hcand_results

    #print("ckajsndckjxkas kjdcnsdakj ")
    events, hcandprod_results = self[higgscandprod](events, hcand_array)
    results += hcandprod_results
    
    #from IPython import embed; embed()

    # create process ids
    events = self[process_ids](events, **kwargs)

    #print("ckajsndckjxkas kjdcnsdakj qwb cjhxb ashjbdjhabxhjk")

    #from IPython import embed; embed()
    
    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel
    
    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)


    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": event_sel,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.event),
        }
        group_map = {
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
            # per channel
            "channel": {
                "values": events.channel_id,
                "mask_fn": (lambda v: events.channel_id == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )
    """
    events, results = self[custom_increment_stats]( 
        events,
        results,
        stats,
    )
    """
    #from IPython import embed; embed()
    #1/0
    return events, results
