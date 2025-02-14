#!/bin/bash
source ./common.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        --config $config
        --processes $processes
        --datasets $datasets
        --version $version
        --categories 'mutau_signal_reg'
        --cf.CalibrateEvents-workflow $workflow
        --cf.SelectEvents-workflow $workflow
        --cf.ReduceEvents-workflow $workflow
        --cf.MergeReducedEvents-workflow $workflow
        --variables 'mutau_lep0_pt,mutau_lep0_eta,mutau_lep0_phi,mutau_lep0_ip_sig,'`
                    `'mutau_lep1_pt,mutau_lep1_eta,mutau_lep1_phi,mutau_lep1_mass,mutau_lep1_decayModePNet,mutau_lep1_decayMode,'`
                    `'mutau_mt,mutau_mvis,mutau_delta_r,mutau_pt,puppi_met_pt,puppi_met_phi'
        --file-types pdf
	    --hist-hooks qcd_hook
        --general-settings "cms-label=pw"
        "${@:2}"
    )
echo law run cf.PlotVariables1D "${args[@]}"
law run cf.PlotVariables1D "${args[@]}" 



