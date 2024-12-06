#!/bin/bash

set_common_vars() {

version="dev"
    
categories_mutau="mutau_signal_reg" 
variables_mutau='mutau_lep0_pt,mutau_lep0_eta,mutau_lep0_phi,mutau_lep0_ip_sig,mutau_lep1_pt,mutau_lep1_eta,mutau_lep1_phi,mutau_lep1_mass,mutau_lep1_decayModePNet,mutau_lep1_decayMode,mutau_mt,mutau_mvis,mutau_delta_r,mutau_pt,puppi_met_pt,puppi_met_phi'

categories_etau="etau_signal_reg" 
variables_etau='etau_lep0_pt,etau_lep0_eta,etau_lep0_phi,etau_lep0_ip_sig,etau_lep1_pt,etau_lep1_eta,etau_lep1_phi,etau_lep1_mass,etau_lep1_decayModePNet,etau_lep1_decayMode,etau_mt,etau_mvis,etau_delta_r,etau_pt,puppi_met_pt,puppi_met_phi'

data_e_2022preEE='data_e_C,data_e_D,'
data_mu_2022preEE='data_singlemu_C,data_mu_C,data_mu_D,'

data_e_2022postEE='data_e_E,data_e_F,data_e_G,'
data_mu_2022postEE='data_mu_E,data_mu_F,data_mu_G,'

bkg_ewk_2022preEE='wj_incl,ww,wz,zz,dy_incl,'
bkg_top_2022preEE='st_twchannel_t_sl,st_twchannel_t_dl,st_twchannel_tbar_sl,st_twchannel_tbar_dl,st_tchannel_tbar,st_tchannel_t,st_schannel_t_lep,st_schannel_tbar_lep,'
bkg_ttbar_2022preEE='tt_sl,tt_dl,tt_fh'

bkg_ewk_2022postEE='wj_incl_madgraph,ww,wz,zz,dy_lep_madgraph,'
bkg_top_2022postEE='st_twchannel_t_sl,st_twchannel_t_dl,st_twchannel_tbar_sl,st_twchannel_tbar_dl,st_tchannel_tbar,st_tchannel_t,st_schannel_t_lep,st_schannel_tbar_lep,'
bkg_ttbar_2022postEE='tt_sl,tt_dl,tt_fh'

case $1 in
##############################
####### 2022preEE ############
##############################
    "run3_2022preEE_etau_lim")
        config="run3_2022_preEE_etau_limited"	
        datasets='dy_incl'
        processes='dy_z2mumu,dy_z2ee,dy_lep,vv'
	categories=$categories_etau
	variables=$variables_etau
        workflow='local'
    ;;
    "run3_2022preEE_mutau_lim")
        config="run3_2022_preEE_mutau_limited"	
        datasets='dy_incl'
        processes='dy_z2mumu,dy_z2ee,dy_lep'
	categories=$categories_mutau
	variables=$variables_mutau
        workflow='local'
    ;;
    "run3_2022preEE_etau")
        config="run3_2022_preEE_etau"
        data=$data_e_2022preEE
        bkg_ewk=$bkg_ewk_2022preEE
        bkg_top=$bkg_top_2022preEE
        bkg_ttbar=$bkg_ttbar_2022preEE
        datasets="$data$bkg_ewk$bkg_top$bkg_ttbar"
        processes='dy_z2tautau,dy_z2mumu,dy_z2ee,vv,tt,st,wj,data'
	categories=$categories_etau
	variables=$variables_etau
        workflow='htcondor'
    ;;
    "run3_2022preEE_mutau")
        config="run3_2022_preEE_mutau"
        data=$data_mu_2022preEE
        bkg_ewk=$bkg_ewk_2022preEE
        bkg_top=$bkg_top_2022preEE
        bkg_ttbar=$bkg_ttbar_2022preEE
        datasets="$data$bkg_ewk$bkg_top$bkg_ttbar"
        processes='dy_z2tautau,dy_z2mumu,dy_z2ee,vv,tt,st,wj,data'
	categories=$categories_mutau
	variables=$variables_mutau
        workflow='htcondor'
     ;;
###############################
######## 2022postEE ###########
###############################    
    "run3_2022postEE_etau_lim")
        config="run3_2022_postEE_etau_limited"	
        datasets='dy_lep_madgraph' 
        processes='dy_z2mumu,dy_z2ee,dy_lep' 
	categories=$categories_etau
	variables=$variables_etau
        workflow='local'
    ;;
    "run3_2022postEE_mutau_lim")
        config="run3_2022_postEE_mutau_limited"	
        datasets='dy_lep_madgraph' 
        processes='dy_z2mumu,dy_z2ee,dy_lep' 
	categories=$categories_mutau
	variables=$variables_mutau
        workflow='local'
    ;;
    "run3_2022postEE_etau")
        config="run3_2022_postEE_etau"
        data=$data_e_2022postEE
        bkg_ewk=$bkg_ewk_2022postEE
        bkg_top=$bkg_top_2022postEE
        bkg_ttbar=$bkg_ttbar_2022postEE
	#        datasets="$data$bkg_ewk$bkg_top$bkg_ttbar"
	datasets=$data
        processes='dy_z2tautau,dy_z2mumu,dy_z2ee,vv,tt,st,wj,data'
	categories=$categories_etau
	variables=$variables_etau
	workflow='htcondor'
    ;;
    "run3_2022postEE_mutau")
        config="run3_2022_postEE_mutau"
        data=$data_mu_2022postEE
        bkg_ewk=$bkg_ewk_2022postEE
        bkg_top=$bkg_top_2022postEE
        bkg_ttbar=$bkg_ttbar_2022postEE
	datasets="$data$bkg_ewk$bkg_top$bkg_ttbar"
        processes='dy_z2tautau,dy_z2mumu,dy_z2ee,vv,tt,st,wj,data'
	categories=$categories_mutau
	variables=$variables_mutau
	workflow='htcondor'
    ;;
    *)
    echo "Unknown run argument!"
    exit
    ;;
esac
}
