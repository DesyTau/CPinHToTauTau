#!/bin/bash
source ./common.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        --config $config
        --processes $processes
        --version $version
        --datasets $datasets
        --cf.CalibrateEvents-workflow $workflow
        --cf.SelectEvents-workflow $workflow
        --cf.ReduceEvents-workflow $workflow
        --cf.MergeSelectionMasks-workflow local
        --selector-steps trigger,met_filter,has_at_least_2_leptons,single_hcand,dilepton_veto,extra_lepton_veto,decay_prods_are_ok
        "${@:2}"
    )
echo run cf.PlotCutflow "${args[@]}"
law run cf.PlotCutflow "${args[@]}"