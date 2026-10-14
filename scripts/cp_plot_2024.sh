#!/bin/bash
source ./cp_samples_2024_nanov15.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        --configs $configs
        --processes $processes
        --datasets  $datasets
        --version $version
        
        --categories 'cat_mutau_sr__fake_incl__bdt_incl,\
        cat_mutau_sr__fake_incl__bdt_incl__tau2pi,\
        cat_mutau_sr__fake_incl__bdt_incl__tau2rho,\
        cat_mutau_sr__fake_incl__bdt_incl__tau2a1,\
        cat_mutau_sr__fake_incl__bdt_incl__tau2a1_3pr'
        
        # cat_mutau_abcd_ar__fake_incl__bdt_incl__tau2pi,\
        # cat_mutau_abcd_ar__fake_incl__bdt_incl__tau2rho,\
        # cat_mutau_abcd_ar__fake_incl__bdt_incl__tau2a1,\
        # cat_mutau_abcd_ar__fake_incl__bdt_incl__tau2a1_3pr'
        #`'cat_mutau_sr__fake_incl__tau2pi,cat_mutau_sr__fake_incl__tau2rho,'`
        #`'cat_mutau_sr__fake_incl__tau2a1,cat_mutau_sr__fake_incl__tau2a1_3pr'`
        # 'cat_mutau_sr__fake_incl,cat_mutau_sr__fake_incl__tau2pi,cat_mutau_sr__fake_incl__tau2rho,'`
        #`'cat_mutau_sr__fake_incl__tau2a1,cat_mutau_sr__fake_incl__tau2a1_3pr' 
        
        --cf.CalibrateEvents-workflow $workflow
        --cf.CalibrateEvents-htcondor-memory 3GB 
        --cf.CalibrateEvents-max-runtime 2
        --cf.CalibrateEvents-retries 2

        --cf.SelectEvents-workflow $workflow
        --cf.SelectEvents-htcondor-memory 5GB
        --cf.SelectEvents-max-runtime 4
        --cf.SelectEvents-retries 1

        --cf.ReduceEvents-workflow $workflow
        #--cf.ReduceEvents-tasks-per-job 3 
        --cf.ReduceEvents-htcondor-memory 5GB
        --cf.ReduceEvents-max-runtime 2

        --cf.MergeReducedEvents-workflow $workflow
        --cf.MergeReducedEvents-retries 2

        --cf.ProduceColumns-workflow $workflow
        --cf.ProduceColumns-htcondor-memory 10GB
        --cf.ProduceColumns-retries 1

        --cf.CreateHistograms-workflow $workflow
        
        --cf.MergeHistograms-version merge_check
        --cf.MergeHistograms-workflow local
        
        --cf.PlotVariables1D-version merge_check
       
        # --pilot True 

        --variables 'mutau_lep0_pt,mutau_lep0_eta,mutau_lep0_phi,mutau_lep0_mass,mutau_lep0_iso,mutau_lep1_pt,mutau_lep1_eta,mutau_lep1_phi,mutau_lep1_mass,mutau_lep1_iso,mutau_mvis,mutau_mt,mutau_delta_r,puppi_met_pt,puppi_met_phi,mutau_pt_vis,leading_jet_pt,subleading_jet_pt,n_j'
        
    --file-types pdf,png

        #--hist-hooks qcd,incl,order #ff_method_dr_closure_test,qcd,incl,ff_method
        --hist-hooks ff_method,incl,order #ff_method_dr_closure_test,qcd,incl,ff_method
        # 1. ff_method: general fake factor method that requires ff weights to be present at the events tree
        # 2. closure_test: Calclulate fake contribution and apply it to the dr_num regions for the closure tests
        # 3. good_old_abcd: estimates QCD contribution by taking events from same sign region and transfer factors from inv. lep iso

        #--skip-ratio
        --general-settings "cms-label=pw" #,yscale=log" # 
        --process-settings "h_ggf_htt_cpo_prod_sm,unstack,scale=stack,color=#28348e:h_ggf_htt_mm_prod_sm,unstack,scale=stack,color=#2b663c:h_ggf_htt_sm_prod_sm,unstack,scale=stack,color=#d62839:h_vbf_htt_cpo,unstack,scale=stack,color=#1f77b4:h_vbf_htt_sm,unstack,scale=stack,color=#ff7f0e:h_vbf_htt_mm,unstack,scale=stack,color=#2ca02c"
        "${@:2}"
    )

echo law run cf.PlotVariables1D --workers 10 "${args[@]}" 
law run cf.PlotVariables1D --workers 10 "${args[@]}" 