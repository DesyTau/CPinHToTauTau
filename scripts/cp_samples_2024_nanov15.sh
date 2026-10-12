#!/bin/bash

set_common_vars() {

version="merge_check"

data_mu_2024='data_mu_C,data_mu_D,data_mu_E,data_mu_F,data_mu_G,data_mu_H,data_mu_I,'

bkg_dy='DYto2E_MLL_50_0J_amcatnloFXFX,DYto2E_MLL_50_1J_amcatnloFXFX,DYto2E_MLL_50_2J_amcatnloFXFX,DYto2Mu_MLL_50_0J_amcatnloFXFX,DYto2Mu_MLL_50_1J_amcatnloFXFX,DYto2Mu_MLL_50_2J_amcatnloFXFX,DYto2Tau_MLL_50_0J_amcatnloFXFX,DYto2Tau_MLL_50_1J_amcatnloFXFX,DYto2Tau_MLL_50_2J_amcatnloFXFX,'
bkg_wj='WtoLNu_1J_madgraphMLM,WtoLNu_2J_madgraphMLM,WtoLNu_3J_madgraphMLM,WtoLNu_4J_madgraphMLM,'
bkg_vv='WW,WZ,ZZ,'
bkg_ttbar='TTto2L2Nu,TTtoLNu2Q,TTto4Q,'
bkg_top='TBbarQto2Q,TbarBQto2Q,TBbarQtoLNu,TbarBQtoLNu,TWminusto2L2Nu,TbarWplusto2L2Nu,TWminustoLNu2Q,TbarWplustoLNu2Q,'

signal='h_ggf_htt_sm_prod_sm_filtered,h_ggf_htt_mm_prod_sm_filtered,h_ggf_htt_cpo_prod_sm_filtered,h_vbf_htt_sm_filtered' #zh_htt_sm_filtered,wph_htt_sm_filtered,wmh_htt_sm_filtered,'

signal_procs='h_ggf_htt_sm_prod_sm,h_ggf_htt_mm_prod_sm,h_ggf_htt_cpo_prod_sm,h_vbf_htt_sm,' #zh_htt_sm,wph_htt_sm,wmh_htt_sm,h_vbf_htt_sm,h_vbf_htt_mm,h_vbf_htt_cpo'

case $1 in

# ##############################
# ########### 2024 #############
# ##############################
"2024_mt")
        configs="run3_2024_mutau"
        bkgs="$bkg_wj$bkg_dy$bkg_vv$bkg_top$bkg_ttbar$signal"
        datasets="$data_mu_2024${bkgs}"
        processes="$signal_procs,dy_ee_m50,dy_mumu_m50,dy_tt_m50,vv,tt,st,wj,data"
        workflow='htcondor'
      ;;
"run3_2024_emu_lim")
        config="run3_2024_emu_limited"
        data_sample="data_egamma_C"
        dy_sample="DYto2Tau_MLL_50_0J_amcatnloFXFX"
        signal_sample="ggphi_phitt_100"
        datasets="${data_sample},${dy_sample},${signal_sample},"
        processes="data,dy_tt_m50,ggphi_phitt_100"
        categories="cat_emu_sr"
        variables=$variables_emu
        workflow="local"
    ;;
 
    *)
    echo "Unknown run argument!"
    exit

esac
}
