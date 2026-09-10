#!/bin/bash
set -euo pipefail

source ./common_run3_MSSM.sh

set_common_vars "$1"

version="${TEST_VERSION:-datacard_shapes_batch_xgb_test}"

extra_args=("${@:2}")


# ============================================================================
# Mass points
# ============================================================================

mapfile -t masses < <(
    python - <<'PY'
from MSSM_H_tt.config.mass_points import read_bdt_masses

for mass in read_bdt_masses():
    print(mass)
PY
)


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


# ============================================================================
# Common arguments
# ============================================================================

common_args=(
    --configs "$config"
    --processes "$processes"
    --datasets "$datasets"

    --version "$version"

    --shift-sources "$shift_sources_csv"

    --producers "main_common,bdt_card_all"

    --file-types png
    --general-settings "cms-label=pw,yscale=log"

    --workflow htcondor
    --poll-interval 5m
    --pilot True

    "${extra_args[@]}"
)


# ============================================================================
# Plot function
#
# IMPORTANT:
# only one category is given to each call, so there is no unwanted
# category x variable Cartesian product.
# ============================================================================

run_plot() {

    local category="$1"
    local variables="$2"

    echo
    echo "======================================================================"
    echo "Category : $category"
    echo "Variables: $variables"
    echo "======================================================================"
    echo

    law run cf.PlotShiftedVariables1D \
        "${common_args[@]}" \
        --categories "$category" \
        --variables "$variables"
}


# ============================================================================
# Produce exactly the distributions required for every mass
# ============================================================================

for mass in "${masses[@]}"; do

    echo
    echo "######################################################################"
    echo "Mass hypothesis: M${mass}"
    echo "######################################################################"


    # ------------------------------------------------------------------------
    # Signal-like category
    #
    # Only:
    #   D_sig vs Disc_ggphi
    #   D_sig vs Disc_bbphi
    # ------------------------------------------------------------------------

    run_plot \
        "cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}" \
        "bdt_D_sig_vs_Disc_ggphi_M${mass},bdt_D_sig_vs_Disc_bbphi_M${mass}"


    # ------------------------------------------------------------------------
    # DY category
    #
    # Only D_DY
    # ------------------------------------------------------------------------

    run_plot \
        "cat_emu_sr__bdt_dy_M${mass}" \
        "bdt_D_DY_M${mass}"


    # ------------------------------------------------------------------------
    # TT category
    #
    # Only D_TT
    # ------------------------------------------------------------------------

    run_plot \
        "cat_emu_sr__bdt_tt_M${mass}" \
        "bdt_D_TT_M${mass}"

done