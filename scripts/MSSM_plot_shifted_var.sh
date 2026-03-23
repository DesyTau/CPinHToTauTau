#!/bin/bash
source ./common_run3_MSSM.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        # --configs $config
        --config $config
        --processes $processes
        --datasets $datasets
        --version $version
        --categories $categories
        --cf.CalibrateEvents-workflow $workflow
        --cf.SelectEvents-workflow $workflow
        --cf.ReduceEvents-workflow $workflow
        --cf.MergeReducedEvents-workflow $workflow
        --cf.ProduceColumns-workflow $workflow
        --cf.CreateHistograms-workflow $workflow
        --cf.MergeHistograms-workflow local
        --variables $variables
        --shift-sources btag_weight_hf,btag_weight_lf,btag_weight_hfstats1,btag_weight_hfstats2,btag_weight_lfstats1,btag_weight_lfstats2,btag_weight_cferr1,btag_weight_cferr2,jec_Total,jer,Trigger_SF_weight,electron_weight,muon_weight,zpt_weight,pu_weight,top_pt_weight #Unclustered_weight
        --pilot True
        --file-types png
	--hist-hooks qcd
        --hide-stat-errors True
        --draw-total-unc True
        --general-settings "cms-label=pw" #yscale=log,
        --process-settings "dy_lep,color=#FFFF00:h_ggf_htt_100,unstack,scale=100,color=#FF0000:bbh_htt_100,unstack,scale=100,color=#0000FF"
        #"h_ggf_htt_80,unstack,scale=stack,color=#FF0000:h_ggf_htt_100,unstack,scale=stack,color=#0000FF:h_ggf_htt_120,unstack,scale=stack"
        "${@:2}"
    )
echo run cf.PlotShiftedVariables1D "${args[@]}"
law run cf.PlotShiftedVariables1D "${args[@]}"
