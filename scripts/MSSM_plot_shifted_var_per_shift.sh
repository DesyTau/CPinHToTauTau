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
        --variables "puppi_met_pt,puppi_met_pt_recoil_corr"
        --shift-sources unclustered,recoilresp,recoilres,CMS_PS_FSR,CMS_PS_ISR,CMS_Scale_muF,CMS_Scale_muR,btag_weight_hf,btag_weight_lf,btag_weight_hfstats1,btag_weight_hfstats2,btag_weight_lfstats1,btag_weight_lfstats2,btag_weight_cferr1,btag_weight_cferr2,jec_Total,jer,Trigger_SF_weight,electron_weight,muon_weight,zpt_weight,pu_weight,top_pt_weight 
        --file-types png
        --general-settings "cms-label=pw,yscale=log" #yscale=log,
        "${@:2}"
    )
echo run cf.PlotShiftedVariablesPerShift1D "${args[@]}"
law run cf.PlotShiftedVariablesPerShift1D "${args[@]}"