#!/bin/bash


set_common_vars() {

version="all_mass_bdt_test_task_scheduling"


# =============================================================================
# Variables
# =============================================================================

variables_emu_list=(
  "emu_mt_tot"
  "emu_mt_emu"
  "D_zeta"
  "emu_mt_e"
  "emu_mt_mu"
  "N_jets_pT_20_eta_4_7_Tight"
  "leading_jet_eta"
  "subleading_jet_eta"
  "leading_jet_phi"
  "subleading_jet_phi"
  "N_b_jets"
  "leading_jet_pt"
  "subleading_jet_pt"
  "dijet_delta_eta"
  "mjj"
  "leading_b_jet_eta"
  "subleading_b_jet_eta"
  "leading_b_jet_phi"
  "subleading_b_jet_phi"
  "leading_b_jet_pt"
  "subleading_b_jet_pt"
  "di_b_jet_delta_eta"
  "mb_jb_j"
  "emu_lep0_pt"
  "emu_lep0_eta"
  "emu_lep0_phi"
  "emu_lep0_ip_sig"
  "emu_lep1_pt"
  "emu_lep1_eta"
  "emu_lep1_phi"
  "emu_lep1_ip_sig"
  "emu_mvis"
  "emu_delta_r"
  "emu_pt"
  "puppi_met_pt"
  "puppi_met_phi"
  "puppi_met_pt_recoil_corr"
  "pt_H"
  "hcand_emu_fastMTT_mass"
)

variables_emu=$(IFS=,; echo "${variables_emu_list[*]}")


# =============================================================================
# Data
# =============================================================================

data_egamma_2022preEE='data_egamma_C,data_egamma_D,'
data_muoneg_2022preEE='data_muoneg_C,data_muoneg_D,'
data_mu_2022preEE='data_mu_C,data_mu_D,data_singlemu_C,'

data_egamma_2022postEE='data_egamma_E,data_egamma_F,data_egamma_G,'
data_muoneg_2022postEE='data_muoneg_E,data_muoneg_F,data_muoneg_G,'
data_mu_2022postEE='data_mu_E,data_mu_F,data_mu_G,'

data_egamma_2023preBPix='data_egamma_Cv123,data_egamma_Cv4,'
data_muoneg_2023preBPix='data_muoneg_Cv123,data_muoneg_Cv4,'
data_mu_2023preBPix='data_mu_Cv123,data_mu_Cv4,'

data_egamma_2023postBPix='data_egamma_D,'
data_muoneg_2023postBPix='data_muoneg_D,'
data_mu_2023postBPix='data_mu_D,'


# =============================================================================
# Background datasets
# =============================================================================

bkg_dy='DYto2L_M_10to50_amcatnloFXFX,DYto2L_M_50_amcatnloFXFX,DYto2L_M_50_0J_amcatnloFXFX,DYto2L_M_50_1J_amcatnloFXFX,DYto2L_M_50_2J_amcatnloFXFX,DYto2Tau_MLL_50_0J_amcatnloFXFX,DYto2Tau_MLL_50_1J_amcatnloFXFX,DYto2Tau_MLL_50_2J_amcatnloFXFX,'

bkg_dy_no_2Tau_1j='DYto2L_M_10to50_amcatnloFXFX,DYto2L_M_50_amcatnloFXFX,DYto2L_M_50_0J_amcatnloFXFX,DYto2L_M_50_1J_amcatnloFXFX,DYto2L_M_50_2J_amcatnloFXFX,DYto2Tau_MLL_50_0J_amcatnloFXFX,DYto2Tau_MLL_50_2J_amcatnloFXFX,'

bkg_wj='WtoLNu_madgraphMLM,WtoLNu_1J_madgraphMLM,WtoLNu_2J_madgraphMLM,WtoLNu_3J_madgraphMLM,WtoLNu_4J_madgraphMLM,'

bkg_vv='WW,WZ,ZZ,'

bkg_vvv='WWW_4F,WWZ_4F,WZZ,ZZZ,'

bkg_vh_htt='WminusHto2Tau_UncorrelatedDecay_UnFiltered,WplusHto2Tau_UncorrelatedDecay_UnFiltered,ZHto2Tau_UncorrelatedDecay_UnFiltered,'

bkg_higgs='GluGluHto2Tau_UncorrelatedDecay_SM_UnFiltered_ProdAndDecay,VBFHto2Tau_UncorrelatedDecay_UnFiltered,'

bkg_top='TbarWplusto4Q,TWminusto4Q,TbarWplusto2L2Nu,TbarWplustoLNu2Q,TWminusto2L2Nu,TWminustoLNu2Q,'

bkg_ttbar='TTto2L2Nu,TTto4Q,TTtoLNu2Q,'


# Combined background dataset list.
#
# Important:
# signals are deliberately NOT included here.
# The plotting script can append only the two signal datasets of the
# mass currently being plotted.

bkgs="${bkg_dy}${bkg_wj}${bkg_vv}${bkg_vvv}${bkg_vh_htt}${bkg_higgs}${bkg_top}${bkg_ttbar}"


# =============================================================================
# MSSM signal datasets
# =============================================================================

signal='bbphi_phitt_100,ggphi_phitt_100'


signal_bbh='bbphi_phitt_60,bbphi_phitt_65,bbphi_phitt_70,bbphi_phitt_75,bbphi_phitt_80,bbphi_phitt_85,bbphi_phitt_90,bbphi_phitt_95,bbphi_phitt_100,bbphi_phitt_105,bbphi_phitt_110,bbphi_phitt_115,bbphi_phitt_120,bbphi_phitt_125,bbphi_phitt_130,bbphi_phitt_135,bbphi_phitt_140,bbphi_phitt_160,bbphi_phitt_180,bbphi_phitt_200,bbphi_phitt_250,bbphi_phitt_300,bbphi_phitt_350,bbphi_phitt_400,bbphi_phitt_450,bbphi_phitt_500,bbphi_phitt_600,bbphi_phitt_700,bbphi_phitt_800,bbphi_phitt_900,bbphi_phitt_1000,bbphi_phitt_1100,bbphi_phitt_1200,bbphi_phitt_1400,bbphi_phitt_1600,bbphi_phitt_1800,bbphi_phitt_2000,bbphi_phitt_2300,bbphi_phitt_2600,bbphi_phitt_2900,bbphi_phitt_3200,bbphi_phitt_3500,'


signal_ggf='ggphi_phitt_60,ggphi_phitt_65,ggphi_phitt_70,ggphi_phitt_75,ggphi_phitt_80,ggphi_phitt_85,ggphi_phitt_90,ggphi_phitt_95,ggphi_phitt_100,ggphi_phitt_105,ggphi_phitt_110,ggphi_phitt_115,ggphi_phitt_120,ggphi_phitt_125,ggphi_phitt_130,ggphi_phitt_135,ggphi_phitt_140,ggphi_phitt_160,ggphi_phitt_180,ggphi_phitt_200,ggphi_phitt_250,ggphi_phitt_300,ggphi_phitt_350,ggphi_phitt_400,ggphi_phitt_450,ggphi_phitt_500,ggphi_phitt_600,ggphi_phitt_700,ggphi_phitt_800,ggphi_phitt_900,ggphi_phitt_1000,ggphi_phitt_1100,ggphi_phitt_1200,ggphi_phitt_1400,ggphi_phitt_1600,ggphi_phitt_1800,ggphi_phitt_2000,ggphi_phitt_2300,ggphi_phitt_2600,ggphi_phitt_2900,ggphi_phitt_3200,ggphi_phitt_3500'


signal_all="${signal_bbh}${signal_ggf}"


# =============================================================================
# Processes
# =============================================================================

# Background processes only.
#
# This is what the new mass-dependent BDT plotting script should use.
# It then appends:
#
#   bbphi_phitt_${mass}
#   ggphi_phitt_${mass}

processes_bkg="data,dy_lep,dy_tt_m50,h_ggf_htt_sm_prod_sm,st,tt,h_vbf_htt_sm,vh_htt,wj,vv,vvv"


# Full list retained for backward compatibility with existing scripts.

processes_all="${processes_bkg},${signal_all}"


# =============================================================================
# Config selection
# =============================================================================

case $1 in


# =============================================================================
# 2022 + 2023 combined e-mu
# =============================================================================

"22and23_emu")

        config="run3_2022_preEE_emu,run3_2022_postEE_emu,run3_2023_preBPix_emu,run3_2023_postBPix_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2022preEE="${data_egamma_2022preEE}${data_mu_2022preEE}${bkgs}"

        datasets_bkg_2022postEE="${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}"

        datasets_bkg_2023preBPix="${data_egamma_2023preBPix}${data_mu_2023preBPix}${bkgs}"

        datasets_bkg_2023postBPix="${data_egamma_2023postBPix}${data_mu_2023postBPix}${bkgs}"


        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2022preEE}:"
        datasets_bkg="${datasets_bkg}${datasets_bkg_2022postEE}:"
        datasets_bkg="${datasets_bkg}${datasets_bkg_2023preBPix}:"
        datasets_bkg="${datasets_bkg}${datasets_bkg_2023postBPix}"


        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2022preEE}${data_mu_2022preEE}${bkgs}${signal_all}:"

        datasets="${datasets}${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}${signal_all}:"

        datasets="${datasets}${data_egamma_2023preBPix}${data_mu_2023preBPix}${bkgs}${signal_all}:"

        datasets="${datasets}${data_egamma_2023postBPix}${data_mu_2023postBPix}${bkgs}${signal_all}"


        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;
# =============================================================================
# 2022 e-mu
# =============================================================================

"22_emu")

        config="run3_2022_preEE_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2022preEE="${data_egamma_2022preEE}${data_mu_2022preEE}${bkgs}"

        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2022preEE}"

        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2022preEE}${data_mu_2022preEE}${bkgs}${signal_all}"

        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;
# =============================================================================
# 2022EE e-mu
# =============================================================================

"22EE_emu")

        config="run3_2022_postEE_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2022postEE="${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}"

        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2022postEE}"

        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}${signal_all}"

        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;

# =============================================================================
# 2022EE e-mu
# =============================================================================

"22EE_emu")

        config="run3_2022_postEE_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2022postEE="${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}"

        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2022postEE}"

        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2022postEE}${data_mu_2022postEE}${bkgs}${signal_all}"

        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;
# =============================================================================
# 2023 e-mu
# =============================================================================

"23_emu")

        config="run3_2023_preBPix_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2023preBPix="${data_egamma_2023preBPix}${data_mu_2023preBPix}${bkgs}"

        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2023preBPix}"

        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2023preBPix}${data_mu_2023preBPix}${bkgs}${signal_all}"

        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;
# =============================================================================
# 2023BPix e-mu
# =============================================================================

"23BPix_emu")

        config="run3_2023_postBPix_emu"


        # ---------------------------------------------------------------------
        # Background/data datasets only
        #
        # These are exposed separately because the new BDT plotting script
        # should append only:
        #
        #   bbphi_phitt_${mass},ggphi_phitt_${mass}
        #
        # for each configuration.
        # ---------------------------------------------------------------------

        datasets_bkg_2023postBPix="${data_egamma_2023postBPix}${data_mu_2023postBPix}${bkgs}"

        # ---------------------------------------------------------------------
        # Combined background/data dataset string
        # ---------------------------------------------------------------------

        datasets_bkg="${datasets_bkg_2023postBPix}"

        # ---------------------------------------------------------------------
        # Old/full dataset definition
        #
        # Retained for other scripts that still expect all signal datasets.
        # ---------------------------------------------------------------------

        datasets="${data_egamma_2023postBPix}${data_mu_2023postBPix}${bkgs}${signal_all}"

        # ---------------------------------------------------------------------
        # Categories
        #
        # There is deliberately no fixed BDT category here anymore.
        #
        # The plotting script constructs mass-dependent categories:
        #
        #   cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}
        #   cat_emu_sr__bdt_dy_M${mass}
        #   cat_emu_sr__bdt_tt_M${mass}
        # ---------------------------------------------------------------------

        categories=""


        # ---------------------------------------------------------------------
        # Processes
        # ---------------------------------------------------------------------

        processes="$processes_all"

        variables="$variables_emu"

        workflow="htcondor"

        ;;
# =============================================================================
# Unknown option
# =============================================================================

*)

        echo "Unknown configuration option: $1"
        echo
        echo "Available options:"
        echo " 22and23_emu, 22_emu, 22EE_emu, 23_emu, 23BPix_emu"

        return 1

        ;;

esac

}