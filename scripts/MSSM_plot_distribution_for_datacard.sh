#!/bin/bash

source ./common_run3_MSSM.sh

# -------------------------------------------------------------------------
# Common configuration
# -------------------------------------------------------------------------

if [[ -z "$1" ]]; then
    echo "ERROR: no configuration option provided"
    echo
    echo "Usage:"
    echo "  $0 <configuration> [extra law arguments]"
    echo
    echo "Available options:"
    echo "  22and23_emu"
    echo "  22_emu"
    echo "  22EE_emu"
    echo "  23_emu"
    echo "  23BPix_emu"
    exit 1
fi

if ! set_common_vars "$1"; then
    exit 1
fi

extra_args=("${@:2}")

# -------------------------------------------------------------------------
# Analysis channel
#
# All currently supported configurations in this script are e-mu.
# -------------------------------------------------------------------------

channel="emu"


# -------------------------------------------------------------------------
# Common e-mu variables to plot in the inclusive SR
#
# Category:
#   cat_emu_sr
#
# These are passed to PlotDatacardDistributions as the inclusive variables.
# -------------------------------------------------------------------------

variables_emu_list=(
    "emu_mt_tot"
    # "emu_mt_emu"
    # "D_zeta"
    # "emu_mt_e"
    # "emu_mt_mu"
    # "N_jets_pT_20_eta_4_7_Tight"
    # "leading_jet_eta"
    # "subleading_jet_eta"
    # "leading_jet_phi"
    # "subleading_jet_phi"
    # "N_b_jets"
    # "leading_jet_pt"
    # "subleading_jet_pt"
    # "dijet_delta_eta"
    # "mjj"
    # "leading_b_jet_eta"
    # "subleading_b_jet_eta"
    # "leading_b_jet_phi"
    # "subleading_b_jet_phi"
    # "leading_b_jet_pt"
    # "subleading_b_jet_pt"
    # "di_b_jet_delta_eta"
    # "mb_jb_j"
    # "emu_lep0_pt"
    # "emu_lep0_eta"
    # "emu_lep0_phi"
    # "emu_lep0_ip_sig"
    # "emu_lep1_pt"
    # "emu_lep1_eta"
    # "emu_lep1_phi"
    # "emu_lep1_ip_sig"
    # "emu_mvis"
    # "emu_delta_r"
    # "emu_pt"
    # "puppi_met_pt"
    # "puppi_met_phi"
    # "puppi_met_pt_recoil_corr"
    # "pt_H"
    # "hcand_emu_fastMTT_mass"
)

variables_emu="$(
    IFS=,
    echo "${variables_emu_list[*]}"
)"


# ============================================================================
# JEC sources
# ============================================================================

# Common across eras
jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)

# Add the era-dependent sources needed by all requested configs.
IFS=',' read -ra config_list <<< "$config"

for cfg in "${config_list[@]}"; do

    case "$cfg" in
        *2022_preEE*)
            jec_era="2022"
            ;;
        *2022_postEE*)
            jec_era="2022EE"
            ;;
        *2023_preBPix*)
            jec_era="2023"
            ;;
        *2023_postBPix*)
            jec_era="2023BPix"
            ;;
        *)
            echo "ERROR: cannot determine JEC era from config: $cfg"
            exit 1
            ;;
    esac

    jec_sources+=(
        "jec_Regrouped_Absolute_${jec_era}"
        "jec_Regrouped_BBEC1_${jec_era}"
        "jec_Regrouped_EC2_${jec_era}"
        "jec_Regrouped_HF_${jec_era}"
        "jec_Regrouped_RelativeSample_${jec_era}"
    )

done


# ============================================================================
# All systematic sources
# ============================================================================

shift_sources=(
    muon_weight
    electron_weight
    Trigger_SF_weight
    pu_weight
    top_pt_weight
    zpt_weight

    unclustered
    recoilresp
    recoilres

    CMS_PS_ISR
    CMS_PS_FSR
    CMS_Scale_muR
    CMS_Scale_muF

    btag_weight_hf
    btag_weight_lf
    btag_weight_hfstats1
    btag_weight_hfstats2
    btag_weight_lfstats1
    btag_weight_lfstats2
    btag_weight_cferr1
    btag_weight_cferr2

    "${jec_sources[@]}"

    jer
)

shift_sources_csv=$(IFS=,; echo "${shift_sources[*]}")


# -------------------------------------------------------------------------
# Sanity checks
#
# These prevent accidentally starting the wrapper with empty arguments,
# which would otherwise make ColumnFlow resolve defaults or fail later
# during shift-source resolution.
# -------------------------------------------------------------------------

if [[ -z "$config" ]]; then
    echo "ERROR: config is empty"
    exit 1
fi

if [[ -z "$processes" ]]; then
    echo "ERROR: processes is empty"
    exit 1
fi

if [[ -z "$datasets" ]]; then
    echo "ERROR: datasets is empty"
    exit 1
fi

if [[ -z "$shift_sources_csv" ]]; then
    echo "ERROR: shift_sources_csv is empty"
    exit 1
fi


# -------------------------------------------------------------------------
# Schedule all plots in one LAW / Luigi graph
#
# PlotDatacardDistributions creates:
#
#   - one inclusive SR plot task
#
#   - for every configured BDT mass:
#       * merged ggphi + bbphi signal-region plot task
#       * DY-region plot task
#       * TT-region plot task
#
# All child PlotShiftedVariables1D tasks are therefore known to Luigi
# simultaneously. Identical upstream histogram requirements can consequently
# be represented by the same task node instead of being rediscovered by
# separate `law run` invocations.
#
# The BDT mass list itself is read by the Python task from:
#
#   MSSM_H_tt/config/bdt_masses.yaml
#
# through read_bdt_masses(), so the mass list is not duplicated here.
# -------------------------------------------------------------------------

args=(
    --channel "$channel"

    --configs "$config"
    --processes "$processes"
    --datasets "$datasets"
    --version "$version"
    --include-inclusive "True"
    --inclusive-variables "$variables_emu"

    --shift-sources "$shift_sources_csv"

    --file-types "png"

    --general-settings "cms-label=pw"
    --hist-hooks "qcd"

    --workflow "htcondor"
    --workers "8"
    --bypass-branch-requirements "True"
    --poll-interval "5m"
    --pilot "True"

    "${extra_args[@]}"
)


echo
echo "======================================================================"
echo "MSSM datacard distribution production"
echo "======================================================================"
echo "Configuration option: $1"
echo "Configs:              $config"
echo "Channel:              $channel"
echo "Version:              $version"
echo "Inclusive variables:  $variables_emu"
echo
echo "All BDT mass hypotheses will be scheduled in one Luigi graph."
echo "======================================================================"
echo

echo law run MSSM_H_tt.PlotDatacardDistributions "${args[@]}"
echo

law run MSSM_H_tt.PlotDatacardDistributions "${args[@]}"

status=$?

if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: datacard distribution production failed"
    echo
    exit $status
fi