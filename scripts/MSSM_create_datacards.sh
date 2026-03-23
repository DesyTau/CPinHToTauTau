#!/bin/bash
source ./common_run3_MSSM.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
version=desy_dev
args=(
        --config $config

        --cf.CalibrateEvents-version $version
        --cf.CalibrateEvents-workflow $workflow
        
        --cf.SelectEvents-version $version
        --cf.SelectEvents-workflow $workflow

        --cf.ReduceEvents-version $version
        --cf.ReduceEvents-workflow $workflow
        
        --cf.MergeReducedEvents-version $version
        --cf.MergeReducedEvents-workflow $workflow
        
        --cf.MergeSelectionStats-version $version
        --cf.MergeSelectionStats-workflow $workflow

        --cf.ProvideReducedEvents-version $version
        --cf.ProvideReducedEvents-workflow $workflow

        --cf.ProduceColumns-version $version
        --cf.ProduceColumns-workflow $workflow
        
        --cf.MergeHistograms-version $version
        --cf.MergeHistograms-workflow $workflow
        
        --cf.CreateHistograms-version $version
        --cf.CreateHistograms-workflow $workflow
        
        --cf.MergeShiftedHistograms-workflow $workflow
        --cf.MergeShiftedHistograms-version $version
         
        --pilot True
        --version $version
       
        --inference-model MSSM_model
        --hist-hooks qcd
        "${@:2}"
    )
echo law run cf.CreateDatacards "${args[@]}"
law run cf.CreateDatacards "${args[@]}"
