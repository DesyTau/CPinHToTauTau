#!/bin/bash

# Submit one ColumnFlow pipeline stage at a time, optionally splitting
# data, backgrounds, ggphi and bbphi into independent submissions.
#
# Usage:
#   ./MSSM_submit_stage.sh <stage> <config-option> <sample-group> [extra law options]
#
# Examples:
#   ./MSSM_submit_stage.sh calibrate 22and23_emu data
#   ./MSSM_submit_stage.sh calibrate 22and23_emu backgrounds
#   ./MSSM_submit_stage.sh calibrate 22and23_emu ggphi
#   ./MSSM_submit_stage.sh calibrate 22and23_emu bbphi
#
# sample-group:
#   data | backgrounds | signal | ggphi | bbphi
#
# stage:
#   calibrate | select | reduce | merge-reduced |
#   produce | create-hists | merge-hists | merge-shifted

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run3_MSSM.sh"

if (( $# < 3 )); then
    echo "Usage: $0 <stage> <config-option> <sample-group> [extra law options]"
    echo
    echo "Stages:"
    echo "  calibrate"
    echo "  select"
    echo "  reduce"
    echo "  merge-reduced"
    echo "  produce"
    echo "  create-hists"
    echo "  merge-hists"
    echo "  merge-shifted"
    echo
    echo "Sample groups:"
    echo "  data"
    echo "  backgrounds"
    echo "  signal       (ggphi + bbphi)"
    echo "  ggphi"
    echo "  bbphi"
    exit 1
fi

stage="$1"
config_option="$2"
sample_group="$3"
shift 3

set_common_vars "$config_option"

# =============================================================================
# Dataset splitting
# =============================================================================

strip_trailing_comma() {
    local value="$1"
    printf '%s' "${value%,}"
}

case "$sample_group" in

    data)
        # Pass the union of all data dataset names.
        # The ColumnFlow wrapper resolves this list independently for every
        # config and only keeps datasets that actually exist in that config.
        datasets_group="$(
            printf '%s%s%s%s%s%s%s%s' \
                "$data_egamma_2022preEE" \
                "$data_mu_2022preEE" \
                "$data_egamma_2022postEE" \
                "$data_mu_2022postEE" \
                "$data_egamma_2023preBPix" \
                "$data_mu_2023preBPix" \
                "$data_egamma_2023postBPix" \
                "$data_mu_2023postBPix"
        )"
        datasets_group="$(strip_trailing_comma "$datasets_group")"
        ;;

    backgrounds|background|bkg)
        datasets_group="$(strip_trailing_comma "$bkgs")"
        ;;

    signal)
        datasets_group="$(strip_trailing_comma "$signal_all")"
        ;;

    ggphi|ggf)
        datasets_group="$(strip_trailing_comma "$signal_ggf")"
        ;;

    bbphi|bbh)
        datasets_group="$(strip_trailing_comma "$signal_bbh")"
        ;;

    *)
        echo "ERROR: unknown sample group '$sample_group'" >&2
        echo "Allowed: data, backgrounds, signal, ggphi, bbphi" >&2
        exit 1
        ;;
esac
# =============================================================================
# Shift definitions
# =============================================================================

# JEC/JER are the only shifts that require distinct calibrated / selected /
# reduced event streams.
kinematic_shifts="nominal,jec_*,jer_*"

# ProduceColumns additionally has genuine local MET/recoil shifted branches.
producer_shifts="${kinematic_shifts},unclustered_*,recoilresp_*,recoilres_*"

# CreateHistograms / MergeHistograms need every histogram-level variation.
# The wildcard patterns are resolved independently inside each config.
all_hist_shifts="nominal,muon_weight_*,electron_weight_*,top_pt_weight_*,Trigger_SF_weight_*,zpt_weight_*,pu_weight_*,unclustered_*,jec_*,jer_*,CMS_Scale_muR_*,CMS_Scale_muF_*,CMS_PS_ISR_*,CMS_PS_FSR_*,btag_weight_*,recoilresp_*,recoilres_*"

# Shift sources used by MergeShiftedHistograms. Era-dependent JEC names are
# deliberately patterns so that 2022 / 2022EE / 2023 / 2023BPix resolve
# independently for each config.
shift_sources_list=(
    "muon_weight"
    "electron_weight"
    "top_pt_weight"
    "Trigger_SF_weight"
    "zpt_weight"
    "pu_weight"
    "unclustered"

    "jec_Regrouped_Absolute"
    "jec_Regrouped_BBEC1"
    "jec_Regrouped_EC2"
    "jec_Regrouped_HF"
    "jec_Regrouped_RelativeBal"
    "jec_Regrouped_FlavorQCD"

    "jec_Regrouped_Absolute_*"
    "jec_Regrouped_BBEC1_*"
    "jec_Regrouped_EC2_*"
    "jec_Regrouped_HF_*"
    "jec_Regrouped_RelativeSample_*"

    "jer"

    "CMS_Scale_muR"
    "CMS_Scale_muF"
    "CMS_PS_ISR"
    "CMS_PS_FSR"

    "btag_weight_hf"
    "btag_weight_lf"
    "btag_weight_hfstats1"
    "btag_weight_hfstats2"
    "btag_weight_lfstats1"
    "btag_weight_lfstats2"
    "btag_weight_cferr1"
    "btag_weight_cferr2"

    "recoilresp"
    "recoilres"
)
shift_sources=$(IFS=,; echo "${shift_sources_list[*]}")

# Data is processed nominally only.
if [[ "$sample_group" == "data" ]]; then
    kinematic_shifts="nominal"
    producer_shifts="nominal"
    all_hist_shifts="nominal"
fi

# =============================================================================
# Common command fragments
# =============================================================================

common_args=(
    --version "$version"
    --configs "$config"
    --datasets "$datasets_group"
    --poll-interval "5m"
    --pilot "True"
    --parallel-jobs 10
)

# Extra command line options supplied by the user are appended to every stage.
extra_args=("$@")

run_command() {
    echo
    echo "================================================================================"
    echo "Stage        : $stage"
    echo "Config option: $config_option"
    echo "Configs      : $config"
    echo "Sample group : $sample_group"
    echo "================================================================================"
    echo
    printf 'law run'
    printf ' %s' "$@"
    echo
    law run "$@"
}

# =============================================================================
# Stage dispatch
# =============================================================================

case "$stage" in

    calibrate)
        run_command cf.CalibrateEventsWrapper \
            "${common_args[@]}" \
            --calibrator main \
            --shifts "$kinematic_shifts" \
            --cf.CalibrateEvents-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    select)
        run_command cf.SelectEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            --cf.SelectEvents-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    reduce)
        run_command cf.ReduceEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            --cf.ReduceEvents-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    merge-reduced)
        run_command cf.MergeReducedEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            --cf.MergeReducedEvents-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    produce)
        run_command cf.ProduceColumnsWrapper \
            "${common_args[@]}" \
            --producers main \
            --shifts "$producer_shifts" \
            --cf.ProduceColumns-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    create-hists)
        run_command cf.CreateHistogramsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --producers main \
            --variables "$variables" \
            --shifts "$all_hist_shifts" \
            --cf.CreateHistograms-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    merge-hists)
        run_command cf.MergeHistogramsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --producers main \
            --variables "$variables" \
            --shifts "$all_hist_shifts" \
            --cf.MergeHistograms-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    merge-shifted)
        if [[ "$sample_group" == "data" ]]; then
            echo "Data has no shifted histogram merge."
            echo "For data, stage 'merge-hists' with the nominal shift is the final histogram stage."
            exit 0
        fi

        run_command cf.MergeShiftedHistogramsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --producers main \
            --variables "$variables" \
            --shift-sources "$shift_sources" \
            --cf.MergeShiftedHistograms-workflow "$workflow" \
            "${extra_args[@]}"
        ;;

    *)
        echo "ERROR: unknown stage '$stage'" >&2
        echo "Allowed stages:" >&2
        echo "  calibrate, select, reduce, merge-reduced, produce," >&2
        echo "  create-hists, merge-hists, merge-shifted" >&2
        exit 1
        ;;
esac
