#!/bin/bash
source ./common_run3_MSSM.sh #to access set_common_vars() function
#The following function defines config, processes, version and datasets variables
set_common_vars "$1"
args=(
        --configs $config
        # --config $config
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
        --cf.MergeShiftedHistograms-workflow local
        --variables $variables
        --shift-sources unclustered,recoilresp,recoilres,CMS_PS_FSR,CMS_PS_ISR,CMS_Scale_muF,CMS_Scale_muR,btag_weight_hf,btag_weight_lf,btag_weight_hfstats1,btag_weight_hfstats2,btag_weight_lfstats1,btag_weight_lfstats2,btag_weight_cferr1,btag_weight_cferr2,jec_Total,jer,Trigger_SF_weight,electron_weight,muon_weight,zpt_weight,pu_weight,top_pt_weight 
        --pilot True
        --file-types png
	--hist-hooks qcd
        # --hide-stat-errors True
        # --variable-settings "emu_mt_tot,underflow,overflow:emu_mt_emu,underflow,overflow:D_zeta,underflow,overflow:D_zeta_check,underflow,overflow:emu_mt_e,underflow,overflow:emu_mt_mu,underflow,overflow:N_jets_pT_20_eta_4_7_Tight,underflow,overflow:leading_jet_eta,underflow,overflow:subleading_jet_eta,underflow,overflow:leading_jet_phi,underflow,overflow:subleading_jet_phi,underflow,overflow:N_b_jets,underflow,overflow:leading_jet_pt,underflow,overflow:subleading_jet_pt,underflow,overflow:dijet_delta_eta,underflow,overflow:mjj,underflow,overflow:leading_b_jet_eta,underflow,overflow:subleading_b_jet_eta,underflow,overflow:leading_b_jet_phi,underflow,overflow:subleading_b_jet_phi,underflow,overflow:leading_b_jet_pt,underflow,overflow:subleading_b_jet_pt,underflow,overflow:di_b_jet_delta_eta,underflow,overflow:mb_jb_j,underflow,overflow:emu_lep0_pt,underflow,overflow:emu_lep0_eta,underflow,overflow:emu_lep0_phi,underflow,overflow:emu_lep0_ip_sig,underflow,overflow:emu_lep1_pt,underflow,overflow:emu_lep1_eta,underflow,overflow:emu_lep1_phi,underflow,overflow:emu_lep1_ip_sig,underflow,overflow:emu_mvis,underflow,overflow:emu_delta_r,underflow,overflow:emu_pt,underflow,overflow:puppi_met_pt,underflow,overflow:puppi_met_phi,underflow,overflow:pt_H,underflow,overflow:hcand_emu_fastMTT_mass,underflow,overflow"
        --draw-total-unc True
        --general-settings "cms-label=pw" #yscale=log,
        --process-settings "dy_lep,color=#FFFF00" #:ggphi_phitt_100,unstack,scale=1000,color=#FF0000:bbphi_phitt_100,unstack,scale=1000,color=#0000FF"
        # --hist-hooks blind_sr
        "${@:2}"
    )
echo run cf.PlotShiftedVariables1D "${args[@]}"
law run cf.PlotShiftedVariables1D "${args[@]}"
