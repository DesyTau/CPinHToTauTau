#!/bin/bash

source ./common_run3_MSSM.sh  # to access set_common_vars()

# Defines config, processes, version, datasets, variables, categories and workflow.
set_common_vars "$1"

# =============================================================================
# Shift sources
#
# The regrouped JEC sources containing an era suffix are selected with patterns.
# ColumnFlow resolves these patterns independently for each config, e.g.
#
#   2022 preEE   -> jec_Regrouped_Absolute_2022
#   2022 postEE  -> jec_Regrouped_Absolute_2022EE
#   2023 preBPix -> jec_Regrouped_Absolute_2023
#   2023 postBPix-> jec_Regrouped_Absolute_2023BPix
#
# The CSV value is quoted when passed to law so that the shell does not expand '*'.
# =============================================================================

shift_sources_list=(
    "muon_weight"
    "electron_weight"
    "top_pt_weight"
    "Trigger_SF_weight"
    "zpt_weight"
    "pu_weight"
    "unclustered"

    # JEC: common sources
    "jec_Regrouped_Absolute"
    "jec_Regrouped_BBEC1"
    "jec_Regrouped_EC2"
    "jec_Regrouped_HF"
    "jec_Regrouped_RelativeBal"
    "jec_Regrouped_FlavorQCD"

    # JEC: era-dependent sources
    "jec_Regrouped_Absolute_*"
    "jec_Regrouped_BBEC1_*"
    "jec_Regrouped_EC2_*"
    "jec_Regrouped_HF_*"
    "jec_Regrouped_RelativeSample_*"

    "jer"

    # Theory
    "CMS_Scale_muR"
    "CMS_Scale_muF"
    "CMS_PS_ISR"
    "CMS_PS_FSR"

    # b tagging
    "btag_weight_hf"
    "btag_weight_lf"
    "btag_weight_hfstats1"
    "btag_weight_hfstats2"
    "btag_weight_lfstats1"
    "btag_weight_lfstats2"
    "btag_weight_cferr1"
    "btag_weight_cferr2"

    # MET recoil
    "recoilresp"
    "recoilres"
)

shift_sources=$(IFS=,; echo "${shift_sources_list[*]}")

args=(
        --configs $config
        --processes $processes
        --datasets $datasets
        --version $version
        --cf.CalibrateEvents-workflow $workflow
        --cf.SelectEvents-workflow $workflow
        --cf.ReduceEvents-workflow $workflow
        --cf.MergeReducedEvents-workflow $workflow
        --cf.ProduceColumns-workflow $workflow
        --cf.CreateHistograms-workflow $workflow
        --cf.MergeHistograms-workflow $workflow
        --cf.MergeShiftedHistograms-workflow $workflow
        --variables $variables
        --shift-sources "$shift_sources"
        --pilot True
        --file-types png
	--hist-hooks qcd
        # --hide-stat-errors True
        --merge-stat-errors
        # --variable-settings "emu_mt_tot,underflow,overflow:emu_mt_emu,underflow,overflow:D_zeta,underflow,overflow:D_zeta_check,underflow,overflow:emu_mt_e,underflow,overflow:emu_mt_mu,underflow,overflow:N_jets_pT_20_eta_4_7_Tight,underflow,overflow:leading_jet_eta,underflow,overflow:subleading_jet_eta,underflow,overflow:leading_jet_phi,underflow,overflow:subleading_jet_phi,underflow,overflow:N_b_jets,underflow,overflow:leading_jet_pt,underflow,overflow:subleading_jet_pt,underflow,overflow:dijet_delta_eta,underflow,overflow:mjj,underflow,overflow:leading_b_jet_eta,underflow,overflow:subleading_b_jet_eta,underflow,overflow:leading_b_jet_phi,underflow,overflow:subleading_b_jet_phi,underflow,overflow:leading_b_jet_pt,underflow,overflow:subleading_b_jet_pt,underflow,overflow:di_b_jet_delta_eta,underflow,overflow:mb_jb_j,underflow,overflow:emu_lep0_pt,underflow,overflow:emu_lep0_eta,underflow,overflow:emu_lep0_phi,underflow,overflow:emu_lep0_ip_sig,underflow,overflow:emu_lep1_pt,underflow,overflow:emu_lep1_eta,underflow,overflow:emu_lep1_phi,underflow,overflow:emu_lep1_ip_sig,underflow,overflow:emu_mvis,underflow,overflow:emu_delta_r,underflow,overflow:emu_pt,underflow,overflow:puppi_met_pt,underflow,overflow:puppi_met_phi,underflow,overflow:pt_H,underflow,overflow:hcand_emu_fastMTT_mass,underflow,overflow"
        #--draw-total-unc True
        --general-settings "cms-label=pw" #yscale=log, #hide_dat=True
        --process-settings "dy_lep,color=#FFFF00" #:ggphi_phitt_100,unstack,scale=1000,color=#FF0000:bbphi_phitt_100,unstack,scale=1000,color=#0000FF"
        "${@:2}"
    )

# Do not pass an empty --categories value.
if [[ -n "$categories" ]]; then
    args+=(--categories "$categories")
fi

# Append any additional command-line options after the preset ones.
if (( $# > 1 )); then
    args+=("${@:2}")
fi

echo law run cf.PlotShiftedVariables1D "${args[@]}"
law run cf.PlotShiftedVariables1D "${args[@]}"
# #!/bin/bash
# source ./common_run3_MSSM.sh #to access set_common_vars() function
# #The following function defines config, processes, version and datasets variables
# set_common_vars "$1"
# args=(
#         # --configs $config
#         --config $config
#         --processes $processes
#         --datasets $datasets
#         --version $version
#         --categories $categories
#         --cf.CalibrateEvents-workflow $workflow
#         --cf.SelectEvents-workflow $workflow
#         --cf.ReduceEvents-workflow $workflow
#         --cf.MergeReducedEvents-workflow $workflow
#         --cf.ProduceColumns-workflow $workflow
#         --cf.CreateHistograms-workflow $workflow
#         --cf.MergeHistograms-workflow $workflow
#         --cf.MergeShiftedHistograms-workflow $workflow
#         --variables $variables
#         --shift-sources muon_weight,electron_weight,top_pt_weight,Trigger_SF_weight,zpt_weight,pu_weight,unclustered,jec_Regrouped_Absolute,jec_Regrouped_Absolute_2022,jec_Regrouped_BBEC1,jec_Regrouped_BBEC1_2022,jec_Regrouped_EC2,jec_Regrouped_EC2_2022,jec_Regrouped_HF,jec_Regrouped_HF_2022,jec_Regrouped_RelativeBal,jec_Regrouped_RelativeSample_2022,jec_Regrouped_FlavorQCD,jer,CMS_Scale_muR,CMS_Scale_muF,CMS_PS_ISR,CMS_PS_FSR,btag_weight_hf,btag_weight_lf,btag_weight_hfstats1,btag_weight_hfstats2,btag_weight_lfstats1,btag_weight_lfstats2,btag_weight_cferr1,btag_weight_cferr2,recoilresp,recoilres
#         --pilot True
#         --file-types png
# 	--hist-hooks qcd
#         # --hide-stat-errors True
#         --merge-stat-errors
#         # --variable-settings "emu_mt_tot,underflow,overflow:emu_mt_emu,underflow,overflow:D_zeta,underflow,overflow:D_zeta_check,underflow,overflow:emu_mt_e,underflow,overflow:emu_mt_mu,underflow,overflow:N_jets_pT_20_eta_4_7_Tight,underflow,overflow:leading_jet_eta,underflow,overflow:subleading_jet_eta,underflow,overflow:leading_jet_phi,underflow,overflow:subleading_jet_phi,underflow,overflow:N_b_jets,underflow,overflow:leading_jet_pt,underflow,overflow:subleading_jet_pt,underflow,overflow:dijet_delta_eta,underflow,overflow:mjj,underflow,overflow:leading_b_jet_eta,underflow,overflow:subleading_b_jet_eta,underflow,overflow:leading_b_jet_phi,underflow,overflow:subleading_b_jet_phi,underflow,overflow:leading_b_jet_pt,underflow,overflow:subleading_b_jet_pt,underflow,overflow:di_b_jet_delta_eta,underflow,overflow:mb_jb_j,underflow,overflow:emu_lep0_pt,underflow,overflow:emu_lep0_eta,underflow,overflow:emu_lep0_phi,underflow,overflow:emu_lep0_ip_sig,underflow,overflow:emu_lep1_pt,underflow,overflow:emu_lep1_eta,underflow,overflow:emu_lep1_phi,underflow,overflow:emu_lep1_ip_sig,underflow,overflow:emu_mvis,underflow,overflow:emu_delta_r,underflow,overflow:emu_pt,underflow,overflow:puppi_met_pt,underflow,overflow:puppi_met_phi,underflow,overflow:pt_H,underflow,overflow:hcand_emu_fastMTT_mass,underflow,overflow"
#         #--draw-total-unc True
#         --general-settings "cms-label=pw" #yscale=log, #hide_dat=True
#         --process-settings "dy_lep,color=#FFFF00" #:ggphi_phitt_100,unstack,scale=1000,color=#FF0000:bbphi_phitt_100,unstack,scale=1000,color=#0000FF"
#         "${@:2}"
#     )
# echo run cf.PlotShiftedVariables1D "${args[@]}"
# law run cf.PlotShiftedVariables1D "${args[@]}"