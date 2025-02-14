#!/bin/bash
source ./common_run3.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        --config $config
        --processes $processes
        --datasets $datasets
        --version $version
        --categories 'etau_signal_reg_0_bjets'
        --cf.CalibrateEvents-workflow $workflow
        --cf.SelectEvents-workflow $workflow
        --cf.ReduceEvents-workflow $workflow
        --cf.MergeReducedEvents-workflow $workflow
        "${@:2}"
    )
echo law run cf.CreateYieldTable "${args[@]}"
law run cf.CreateYieldTable "${args[@]}" 


#,etau_signal_reg_endcap_tau,etau_signal_reg_barrel_tau,etau_signal_reg_endcap_tau_no_mt,etau_signal_reg_barrel_tau_no_mt'
