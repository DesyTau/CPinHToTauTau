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
    echo "  23preBPix_emu"
    echo "  23postBPix_emu"
    exit 1
fi

if ! set_common_vars "$1"; then
    exit 1
fi

extra_args=("${@:2}")

# -------------------------------------------------------------------------
# Read all BDT mass points directly from the analysis YAML
#
# This avoids duplicating the list here and keeps this script automatically
# synchronized with:
#
#   MSSM_H_tt/config/bdt_masses.yaml
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

mass_file="${REPO_DIR}/MSSM_H_tt/config/bdt_masses.yaml"

if [[ ! -f "$mass_file" ]]; then
    echo "ERROR: mass file not found:"
    echo "  $mass_file"
    exit 1
fi

mapfile -t masses < <(
    awk '
        /^[[:space:]]*-[[:space:]]*[0-9]+[[:space:]]*$/ {
            print $2
        }
    ' "$mass_file"
)

if [[ ${#masses[@]} -eq 0 ]]; then
    echo "ERROR: no BDT masses found in $mass_file"
    exit 1
fi

echo "Found ${#masses[@]} BDT mass points:"
echo "${masses[*]}"


# -------------------------------------------------------------------------
# Common e-mu variables to plot in the inclusive SR
#
# Category:
#   cat_emu_sr
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

# Convert bash array to comma-separated string
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
# Common plotting function
# -------------------------------------------------------------------------

run_plot() {

    local mass="$1"
    local region="$2"
    local category="$3"
    local variables="$4"

    args=(
        --configs "$config"
        --processes "$processes"
        --datasets "$datasets"
        --version "$version"

        --categories "$category"
        --variables "$variables"

        --shift-sources "$shift_sources_csv"

        --file-types "png"

        --general-settings "cms-label=pw"
        --hist-hooks "qcd"
        --workflow "htcondor"
        --workers "100"
        --bypass-branch-requirements "True"
        --poll-interval "5m"
        --pilot "True"

        "${extra_args[@]}"
    )

    echo
    echo "======================================================================"

    if [[ "$mass" == "inclusive" ]]; then
        echo "Mass:      inclusive"
    else
        echo "Mass:      M${mass}"
    fi

    echo "Region:    ${region}"
    echo "Category:  ${category}"
    echo "Variables: ${variables}"
    echo "Version: ${version}"
    echo "======================================================================"
    echo

    echo law run cf.PlotShiftedVariables1D "${args[@]}"

    law run cf.PlotShiftedVariables1D "${args[@]}"

    status=$?

    if [[ $status -ne 0 ]]; then
        echo
        echo "ERROR:"

        if [[ "$mass" == "inclusive" ]]; then
            echo "Plotting failed for inclusive SR"
        else
            echo "Plotting failed for M${mass}, region ${region}"
        fi

        echo
        exit $status
    fi
}


# -------------------------------------------------------------------------
# Plot standard e-mu variables in the inclusive signal region
#
# Category:
#   cat_emu_sr
#
# These variables are independent of the BDT mass hypothesis, so they are
# plotted only once rather than once for every BDT mass.
# -------------------------------------------------------------------------

run_plot \
    "inclusive" \
    "SR" \
    "cat_emu_sr" \
    "$variables_emu"


# -------------------------------------------------------------------------
# Loop over all mass hypotheses
# -------------------------------------------------------------------------

for mass in "${masses[@]}"; do

    # ---------------------------------------------------------------------
    # Merged signal region
    #
    # Category:
    #   cat_emu_sr__bdt_ggphi_and_bbphi_M{mass}
    #
    # Datacard variables:
    #   bdt_D_sig_vs_Disc_ggphi_M{mass}
    #   bdt_D_sig_vs_Disc_bbphi_M{mass}
    # ---------------------------------------------------------------------

    signal_category="cat_emu_sr__bdt_ggphi_and_bbphi_M${mass}"

    signal_variables="bdt_D_sig_vs_Disc_ggphi_M${mass},bdt_D_sig_vs_Disc_bbphi_M${mass}"

    run_plot \
        "$mass" \
        "signal" \
        "$signal_category" \
        "$signal_variables"


    # ---------------------------------------------------------------------
    # DY region
    #
    # Category:
    #   cat_emu_sr__bdt_dy_M{mass}
    #
    # Datacard variable:
    #   bdt_D_DY_M{mass}
    # ---------------------------------------------------------------------

    dy_category="cat_emu_sr__bdt_dy_M${mass}"

    dy_variable="bdt_D_DY_M${mass}"

    run_plot \
        "$mass" \
        "DY" \
        "$dy_category" \
        "$dy_variable"


    # ---------------------------------------------------------------------
    # TT region
    #
    # Category:
    #   cat_emu_sr__bdt_tt_M{mass}
    #
    # Datacard variable:
    #   bdt_D_TT_M{mass}
    # ---------------------------------------------------------------------

    tt_category="cat_emu_sr__bdt_tt_M${mass}"

    tt_variable="bdt_D_TT_M${mass}"

    run_plot \
        "$mass" \
        "TT" \
        "$tt_category" \
        "$tt_variable"

done