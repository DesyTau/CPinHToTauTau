#!/bin/bash
# Submit one ColumnFlow pipeline stage at a time.
#
# Usage:
#
#   ./MSSM_submit_stage.sh <config-option> <sample-group> <stage> [extra LAW options]
#
#   ./MSSM_submit_stage.sh 22_emu wjets produce --workers 1 --workflow htcondor
#
# Background/data production defaults to nominal and uses the configured
# histogram producer. Downstream stages check existing event inputs before
# running. The previous stage-first order works too.
# Plot categories can be selected with --category NAME or --categories NAME.
#
# Stages:
#
#   calibrate
#   select
#   reduce
#   merge-reduced
#   check-hist-inputs
#   produce
#   create-hists
#   merge-hists
#   merge-shifted
#   plot1d
#   plot1d-shifted
#
# Sample groups:
#
#   data
#   backgrounds
#   DY
#   tt
#   singlet
#   wjets
#   other_bkgs
#   signal
#   ggphi
#   bbphi
#   datacard
#
# =============================================================================
# Workflow selection
# =============================================================================
#
# Default:
#
#   all task families -> htcondor
#
# Set everything local:
#
#   --workflow local
#
# Set individual task families:
#
#   --calibrate-workflow local|htcondor
#   --select-workflow local|htcondor
#   --reduce-workflow local|htcondor
#   --merge-reduced-workflow local|htcondor
#   --produce-workflow local|htcondor
#   --create-hists-workflow local|htcondor
#   --merge-hists-workflow local|htcondor
#   --merge-shifted-workflow local|htcondor
#
# Native ColumnFlow names are accepted too:
#
#   --cf.CalibrateEvents-workflow local
#   --cf.SelectEvents-workflow local
#   --cf.ReduceEvents-workflow local
#   --cf.MergeReducedEvents-workflow local
#   --cf.ProduceColumns-workflow local
#   --cf.CreateHistograms-workflow local
#   --cf.MergeHistograms-workflow local
#   --cf.MergeShiftedHistograms-workflow local
#
# =============================================================================
# Parallelism
# =============================================================================
#
# HTCondor:
#
#   --parallel-jobs 30
#
# Local:
#
#   --workers 5
#
# Example:
#
#   ./MSSM_submit_stage.sh \
#       22and23_emu DY plot1d-shifted \
#       --variables bdt_D_sig_vs_Disc_ggphi_M100 \
#       --categories cat_emu_sr__bdt_ggphi_and_bbphi_M100 \
#       --producers main \
#       --create-hists-workflow local \
#       --merge-hists-workflow local \
#       --merge-shifted-workflow local \
#       --workers 5
set -e
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# =============================================================================
# Arguments
# =============================================================================
if (( $# < 3 )); then
    echo "Usage: $0 <config-option> <sample-group> <stage> [extra law options]"
    echo
    echo "Stages:"
    echo "  calibrate"
    echo "  select"
    echo "  reduce"
    echo "  merge-reduced"
    echo "  check-hist-inputs"
    echo "  produce"
    echo "  create-hists"
    echo "  merge-hists"
    echo "  merge-shifted"
    echo "  plot1d"
    echo "  plot1d-shifted"
    echo
    echo "Sample groups:"
    echo "  data"
    echo "  backgrounds"
    echo "  DY"
    echo "  tt"
    echo "  singlet"
    echo "  wjets"
    echo "  other_bkgs"
    echo "  signal"
    echo "  ggphi"
    echo "  bbphi"
    echo "  datacard"
    exit 1
fi
case "$1" in
    calibrate|select|reduce|merge-reduced|check-hist-inputs|produce|create-hists|merge-hists|merge-shifted|plot1d|plot1d-shifted)
        stage="$1"
        config_option="$2"
        sample_group="$3"
        ;;
    *)
        config_option="$1"
        sample_group="$2"
        stage="$3"
        ;;
esac
shift 3
# =============================================================================
# Datacard positional variable
# =============================================================================
datacard_variable=""
if [[ "$sample_group" == "datacard" ]]; then
    case "$stage" in
        plot1d|plot1d-shifted)
            if (( $# < 1 )) || [[ "$1" == -* ]]; then
                echo "ERROR: sample group 'datacard' requires the final BDT variable." >&2
                echo "Example:" >&2
                echo "  $0 22and23_emu datacard plot1d bdt_D_DY_M100" >&2
                exit 1
            fi
            datacard_variable="$1"
            shift
            ;;
        *)
            echo "ERROR: sample group 'datacard' is only valid for plot1d and plot1d-shifted." >&2
            exit 1
            ;;
    esac
fi
# =============================================================================
# Workflow defaults
# =============================================================================
default_workflow="htcondor"
pilot="False"
signal_masses=""
parallel_masses=4
calibrate_workflow=""
select_workflow=""
reduce_workflow=""
merge_reduced_workflow=""
produce_workflow=""
create_hists_workflow=""
merge_hists_workflow=""
merge_shifted_workflow=""
parallel_jobs=10
validate_workflow() {
    local value="$1"
    local option="$2"
    case "$value" in
        local|htcondor)
            ;;
        *)
            echo "ERROR: invalid workflow '$value' for $option." >&2
            echo "Allowed values: local, htcondor" >&2
            exit 1
            ;;
    esac
}
validate_pilot() {
    case "$1" in
        True|true|1) pilot="True" ;;
        False|false|0) pilot="False" ;;
        *) echo "ERROR: --pilot requires True or False." >&2; exit 1 ;;
    esac
}
# =============================================================================
# Parse script-specific options
# =============================================================================
raw_extra_args=("$@")
extra_args=()
histogram_bdt_masses=""
i=0
while (( i < ${#raw_extra_args[@]} )); do
    arg="${raw_extra_args[$i]}"
    case "$arg" in
        # Normalize the friendly singular spelling before resolving plot defaults.
        --category)
            if (( i + 1 >= ${#raw_extra_args[@]} )) ||
               [[ -z "${raw_extra_args[$((i + 1))]}" || "${raw_extra_args[$((i + 1))]}" == -* ]]; then
                echo "ERROR: --category requires a category name." >&2
                exit 1
            fi
            extra_args+=(--categories "${raw_extra_args[$((i + 1))]}")
            ((i += 2))
            ;;
        --category=*)
            if [[ -z "${arg#*=}" || "${arg#*=}" == -* ]]; then
                echo "ERROR: --category requires a category name." >&2
                exit 1
            fi
            extra_args+=("--categories=${arg#*=}")
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # Parallel jobs
        # ---------------------------------------------------------------------
        --parallel-jobs)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: --parallel-jobs requires an integer." >&2
                exit 1
            fi
            parallel_jobs="${raw_extra_args[$((i + 1))]}"
            if ! [[ "$parallel_jobs" =~ ^[0-9]+$ ]]; then
                echo "ERROR: invalid --parallel-jobs '$parallel_jobs'." >&2
                exit 1
            fi
            ((i += 2))
            ;;
        --parallel-jobs=*)
            parallel_jobs="${arg#*=}"
            if ! [[ "$parallel_jobs" =~ ^[0-9]+$ ]]; then
                echo "ERROR: invalid --parallel-jobs '$parallel_jobs'." >&2
                exit 1
            fi
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # Global workflow
        # ---------------------------------------------------------------------
        --workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: --workflow requires local or htcondor." >&2
                exit 1
            fi
            default_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$default_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --workflow=*)
            default_workflow="${arg#*=}"
            validate_workflow \
                "$default_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # CalibrateEvents
        # ---------------------------------------------------------------------
        --calibrate-workflow|--cf.CalibrateEvents-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            calibrate_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$calibrate_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --calibrate-workflow=*|--cf.CalibrateEvents-workflow=*)
            calibrate_workflow="${arg#*=}"
            validate_workflow \
                "$calibrate_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # SelectEvents
        # ---------------------------------------------------------------------
        --select-workflow|--cf.SelectEvents-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            select_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$select_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --select-workflow=*|--cf.SelectEvents-workflow=*)
            select_workflow="${arg#*=}"
            validate_workflow \
                "$select_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # ReduceEvents
        # ---------------------------------------------------------------------
        --reduce-workflow|--cf.ReduceEvents-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            reduce_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$reduce_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --reduce-workflow=*|--cf.ReduceEvents-workflow=*)
            reduce_workflow="${arg#*=}"
            validate_workflow \
                "$reduce_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # MergeReducedEvents
        # ---------------------------------------------------------------------
        --merge-reduced-workflow|--cf.MergeReducedEvents-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            merge_reduced_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$merge_reduced_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --merge-reduced-workflow=*|--cf.MergeReducedEvents-workflow=*)
            merge_reduced_workflow="${arg#*=}"
            validate_workflow \
                "$merge_reduced_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # ProduceColumns
        # ---------------------------------------------------------------------
        --produce-workflow|--cf.ProduceColumns-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            produce_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$produce_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --produce-workflow=*|--cf.ProduceColumns-workflow=*)
            produce_workflow="${arg#*=}"
            validate_workflow \
                "$produce_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # CreateHistograms
        # ---------------------------------------------------------------------
        --create-hists-workflow|--cf.CreateHistograms-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            create_hists_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$create_hists_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --create-hists-workflow=*|--cf.CreateHistograms-workflow=*)
            create_hists_workflow="${arg#*=}"
            validate_workflow \
                "$create_hists_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # MergeHistograms
        # ---------------------------------------------------------------------
        --merge-hists-workflow|--cf.MergeHistograms-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            merge_hists_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$merge_hists_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --merge-hists-workflow=*|--cf.MergeHistograms-workflow=*)
            merge_hists_workflow="${arg#*=}"
            validate_workflow \
                "$merge_hists_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # MergeShiftedHistograms
        # ---------------------------------------------------------------------
        --merge-shifted-workflow|--cf.MergeShiftedHistograms-workflow)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: $arg requires local or htcondor." >&2
                exit 1
            fi
            merge_shifted_workflow="${raw_extra_args[$((i + 1))]}"
            validate_workflow \
                "$merge_shifted_workflow" \
                "$arg"
            ((i += 2))
            ;;
        --merge-shifted-workflow=*|--cf.MergeShiftedHistograms-workflow=*)
            merge_shifted_workflow="${arg#*=}"
            validate_workflow \
                "$merge_shifted_workflow" \
                "${arg%%=*}"
            ((i += 1))
            ;;
        --histogram-bdt-masses)
            if (( i + 1 >= ${#raw_extra_args[@]} )) ||
               [[ -z "${raw_extra_args[$((i + 1))]}" || "${raw_extra_args[$((i + 1))]}" == -* ]]; then
                echo "ERROR: --histogram-bdt-masses requires mass points or all." >&2
                exit 1
            fi
            histogram_bdt_masses="${raw_extra_args[$((i + 1))]}"
            ((i += 2))
            ;;
        --histogram-bdt-masses=*)
            histogram_bdt_masses="${arg#*=}"
            if [[ -z "$histogram_bdt_masses" ]]; then
                echo "ERROR: --histogram-bdt-masses requires mass points or all." >&2
                exit 1
            fi
            ((i += 1))
            ;;
                    --signal-masses)
            if (( i + 1 >= ${#raw_extra_args[@]} )) ||
               [[ -z "${raw_extra_args[$((i + 1))]}" || "${raw_extra_args[$((i + 1))]}" == -* ]]; then
                echo "ERROR: --signal-masses requires all or a comma-separated mass list." >&2
                exit 1
            fi
            signal_masses="${raw_extra_args[$((i + 1))]}"
            ((i += 2))
            ;;
        --signal-masses=*)
            signal_masses="${arg#*=}"
            [[ -n "$signal_masses" ]] || {
                echo "ERROR: --signal-masses requires all or a comma-separated mass list." >&2
                exit 1
            }
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # Parallel signal masses
        # ---------------------------------------------------------------------
        --parallel-masses)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: --parallel-masses requires a positive integer." >&2
                exit 1
            fi
            parallel_masses="${raw_extra_args[$((i + 1))]}"
            if ! [[ "$parallel_masses" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: --parallel-masses requires a positive integer." >&2
                exit 1
            fi
            ((i += 2))
            ;;
        --parallel-masses=*)
            parallel_masses="${arg#*=}"
            if ! [[ "$parallel_masses" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: --parallel-masses requires a positive integer." >&2
                exit 1
            fi
            ((i += 1))
            ;;
        --pilot)
            if (( i + 1 >= ${#raw_extra_args[@]} )); then
                echo "ERROR: --pilot requires True or False." >&2
                exit 1
            fi
            validate_pilot "${raw_extra_args[$((i + 1))]}"
            ((i += 2))
            ;;
        --pilot=*)
            validate_pilot "${arg#*=}"
            ((i += 1))
            ;;
        # ---------------------------------------------------------------------
        # Everything else -> LAW / ColumnFlow
        # ---------------------------------------------------------------------
        *)
            extra_args+=("$arg")
            ((i += 1))
            ;;
    esac
done
if [[ -n "$histogram_bdt_masses" ]]; then
    case "$stage" in
        create-hists|merge-hists|merge-shifted|plot1d|plot1d-shifted|check-hist-inputs) ;;
        *)
            echo "ERROR: --histogram-bdt-masses applies to histogram stages and plots." >&2
            exit 1
            ;;
    esac
fi
# =============================================================================
# Resolve workflows
# =============================================================================
calibrate_workflow="${calibrate_workflow:-$default_workflow}"
select_workflow="${select_workflow:-$default_workflow}"
reduce_workflow="${reduce_workflow:-$default_workflow}"
merge_reduced_workflow="${merge_reduced_workflow:-$default_workflow}"
produce_workflow="${produce_workflow:-$default_workflow}"
create_hists_workflow="${create_hists_workflow:-$default_workflow}"
merge_hists_workflow="${merge_hists_workflow:-$default_workflow}"
merge_shifted_workflow="${merge_shifted_workflow:-$default_workflow}"
# =============================================================================
# ColumnFlow common setup
# =============================================================================
source "${SCRIPT_DIR}/common_run3_MSSM.sh"
set_common_vars "$config_option"
# =============================================================================
# Helpers
# =============================================================================
strip_trailing_comma() {
    local value="$1"
    printf '%s' "${value%,}"
}
data_for_config() {
    local cfg="$1"
    case "$cfg" in
        run3_2022_preEE_emu*)
            strip_trailing_comma \
                "${data_egamma_2022preEE}${data_mu_2022preEE}"
            ;;
        run3_2022_postEE_emu*)
            strip_trailing_comma \
                "${data_egamma_2022postEE}${data_mu_2022postEE}"
            ;;
        run3_2023_preBPix_emu*)
            strip_trailing_comma \
                "${data_egamma_2023preBPix}${data_mu_2023preBPix}"
            ;;
        run3_2023_postBPix_emu*)
            strip_trailing_comma \
                "${data_egamma_2023postBPix}${data_mu_2023postBPix}"
            ;;
        *)
            echo "ERROR: do not know data datasets for config '$cfg'." >&2
            return 1
            ;;
    esac
}
# Check whether the user explicitly supplied an option.
has_extra_option() {
    local option="$1"
    local arg
    for arg in "${extra_args[@]}"; do
        if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
            return 0
        fi
    done
    return 1
}
# Get the value of an explicitly supplied option.
get_extra_option_value() {
    local option="$1"
    local j
    for ((j = ${#extra_args[@]} - 1; j >= 0; j--)); do
        if [[ "${extra_args[$j]}" == "$option" ]]; then
            if (( j + 1 < ${#extra_args[@]} )); then
                printf '%s' "${extra_args[$((j + 1))]}"
                return 0
            fi
        elif [[ "${extra_args[$j]}" == "$option="* ]]; then
            printf '%s' "${extra_args[$j]#*=}"
            return 0
        fi
    done
    return 1
}
# =============================================================================
# Determine BDT masses requested through --variables
# =============================================================================
#
# Used to automatically restrict:
#
#   signal
#   ggphi
#   bbphi
#
# to the signal samples corresponding to the requested mass(es).
#
# Examples:
#
#   --variables bdt_D_DY_M100
#
# -> requested_bdt_masses=(100)
#
#   --variables bdt_D_DY_M100,bdt_D_DY_M200
#
# -> requested_bdt_masses=(100 200)
#
# =============================================================================
requested_bdt_masses=()
requested_variables="$(
    get_extra_option_value \
        --variables \
        || true
)"
if [[ -n "$requested_variables" ]]; then
    IFS=',' read -r -a requested_variable_array \
        <<< "$requested_variables"
    for var in "${requested_variable_array[@]}"; do
        # Remove accidental spaces.
        var="${var// /}"
        if [[ "$var" =~ ^bdt_(D_sig_vs_Disc_ggphi|D_sig_vs_Disc_bbphi|D_DY|D_TT)_M([0-9]+)$ ]]; then
            mass="${BASH_REMATCH[2]}"
            found=0
            for existing_mass in "${requested_bdt_masses[@]}"; do
                if [[ "$existing_mass" == "$mass" ]]; then
                    found=1
                    break
                fi
            done
            if (( ! found )); then
                requested_bdt_masses+=("$mass")
            fi
        fi
    done
fi
# =============================================================================
# Signal selection helper
# =============================================================================
build_signal_selection() {
    local mode="$1"
    local items=()
    local mass
    # -------------------------------------------------------------------------
    # No mass-dependent BDT variable explicitly requested
    #
    # -> retain the full signal sample group.
    # -------------------------------------------------------------------------
    if (( ${#requested_bdt_masses[@]} == 0 )); then
        case "$mode" in
            signal)
                strip_trailing_comma \
                    "$signal_all"
                ;;
            ggphi)
                strip_trailing_comma \
                    "$signal_ggf"
                ;;
            bbphi)
                strip_trailing_comma \
                    "$signal_bbh"
                ;;
        esac
        return
    fi
    # -------------------------------------------------------------------------
    # Restrict signals to the requested mass(es)
    # -------------------------------------------------------------------------
    for mass in "${requested_bdt_masses[@]}"; do
        case "$mode" in
            signal)
                items+=(
                    "ggphi_phitt_${mass}"
                    "bbphi_phitt_${mass}"
                )
                ;;
            ggphi)
                items+=(
                    "ggphi_phitt_${mass}"
                )
                ;;
            bbphi)
                items+=(
                    "bbphi_phitt_${mass}"
                )
                ;;
        esac
    done
    local IFS=,
    printf '%s' "${items[*]}"
}
selected_signal_all="$(
    build_signal_selection signal
)"
selected_signal_ggphi="$(
    build_signal_selection ggphi
)"
selected_signal_bbphi="$(
    build_signal_selection bbphi
)"
# =============================================================================
# Automatic process selection
# =============================================================================
#
# The sample group now controls plotting processes automatically.
#
# Explicit --processes can still override this later.
# =============================================================================
case "$sample_group" in
    data)
        plot_processes="data"
        ;;
    backgrounds|background|bkg)
        plot_processes="dy_lep,dy_tt_m50,h_ggf_htt_sm_prod_sm,st,tt,h_vbf_htt_sm,vh_htt,wj,vv,vvv"
        ;;
    DY|dy)
        plot_processes="dy_lep,dy_tt_m50"
        ;;
    tt|ttbar)
        plot_processes="tt"
        ;;
    singlet|single_top|single-top)
        plot_processes="st"
        ;;
    wjets|wj|WJets)
        plot_processes="wj"
        ;;
    other_bkgs|other-bkgs|other)
        plot_processes="h_ggf_htt_sm_prod_sm,h_vbf_htt_sm,vh_htt,vv,vvv"
        ;;
    signal)
        plot_processes="$selected_signal_all"
        ;;
    ggphi|ggf)
        plot_processes="$selected_signal_ggphi"
        ;;
    bbphi|bbh)
        plot_processes="$selected_signal_bbphi"
        ;;
    datacard)
        plot_processes=""
        ;;
    *)
        echo "ERROR: unknown sample group '$sample_group'." >&2
        echo "Allowed:" >&2
        echo "  data" >&2
        echo "  backgrounds" >&2
        echo "  DY" >&2
        echo "  tt" >&2
        echo "  singlet" >&2
        echo "  wjets" >&2
        echo "  other_bkgs" >&2
        echo "  signal" >&2
        echo "  ggphi" >&2
        echo "  bbphi" >&2
        echo "  datacard" >&2
        exit 1
        ;;
esac
# =============================================================================
# Datacard specification
# =============================================================================
datacard_discriminant=""
datacard_mass=""
datacard_category=""
datacard_producers=""
datacard_signal_datasets=""
datacard_signal_processes=""
if [[ "$sample_group" == "datacard" ]]; then
    if [[ "$datacard_variable" =~ ^bdt_(D_sig_vs_Disc_ggphi|D_sig_vs_Disc_bbphi|D_DY|D_TT)_M([0-9]+)$ ]]; then
        datacard_discriminant="${BASH_REMATCH[1]}"
        datacard_mass="${BASH_REMATCH[2]}"
    else
        echo "ERROR: unsupported datacard variable '$datacard_variable'." >&2
        exit 1
    fi
    # -------------------------------------------------------------------------
    # Determine analysis channel
    # -------------------------------------------------------------------------
    IFS=',' read -r -a config_array_tmp \
        <<< "$config"
    first_cfg="${config_array_tmp[0]}"
    if [[ "$first_cfg" =~ _(emu|mutau|etau|tautau)($|_) ]]; then
        datacard_channel="${BASH_REMATCH[1]}"
    else
        echo "ERROR: cannot infer channel from '$first_cfg'." >&2
        exit 1
    fi
    # -------------------------------------------------------------------------
    # Variable -> category / signal mapping
    # -------------------------------------------------------------------------
    case "$datacard_discriminant" in
        D_sig_vs_Disc_ggphi)
            datacard_category="cat_${datacard_channel}_sr__bdt_ggphi_and_bbphi_M${datacard_mass}"
            datacard_signal_datasets="ggphi_phitt_${datacard_mass}"
            datacard_signal_processes="ggphi_phitt_${datacard_mass}"
            ;;
        D_sig_vs_Disc_bbphi)
            datacard_category="cat_${datacard_channel}_sr__bdt_ggphi_and_bbphi_M${datacard_mass}"
            datacard_signal_datasets="bbphi_phitt_${datacard_mass}"
            datacard_signal_processes="bbphi_phitt_${datacard_mass}"
            ;;
        D_DY)
            datacard_category="cat_${datacard_channel}_sr__bdt_dy_M${datacard_mass}"
            datacard_signal_datasets="ggphi_phitt_${datacard_mass},bbphi_phitt_${datacard_mass}"
            datacard_signal_processes="ggphi_phitt_${datacard_mass},bbphi_phitt_${datacard_mass}"
            ;;
        D_TT)
            datacard_category="cat_${datacard_channel}_sr__bdt_tt_M${datacard_mass}"
            datacard_signal_datasets="ggphi_phitt_${datacard_mass},bbphi_phitt_${datacard_mass}"
            datacard_signal_processes="ggphi_phitt_${datacard_mass},bbphi_phitt_${datacard_mass}"
            ;;
    esac
    # The required BDT columns are already produced by main.
    datacard_producers="main"
    datacard_background_processes="${processes_bkg:-data,dy_lep,dy_tt_m50,h_ggf_htt_sm_prod_sm,st,tt,h_vbf_htt_sm,vh_htt,wj,vv,vvv}"
    plot_processes="${datacard_background_processes},${datacard_signal_processes}"
fi
# =============================================================================
# Dataset selection
# =============================================================================
IFS=',' read -r -a config_array \
    <<< "$config"
datasets_group=""
for cfg in "${config_array[@]}"; do
    case "$sample_group" in
        data)
            era_datasets="$(
                data_for_config \
                    "$cfg"
            )"
            ;;
        backgrounds|background|bkg)
            era_datasets="$(
                strip_trailing_comma \
                    "$bkgs"
            )"
            ;;
        DY|dy)
            era_datasets="$(
                strip_trailing_comma \
                    "$bkg_dy"
            )"
            ;;
        tt|ttbar)
            era_datasets="$(
                strip_trailing_comma \
                    "$bkg_ttbar"
            )"
            ;;
        singlet|single_top|single-top)
            era_datasets="$(
                strip_trailing_comma \
                    "$bkg_top"
            )"
            ;;
        wjets|wj|WJets)
            era_datasets="$(
                strip_trailing_comma \
                    "$bkg_wj"
            )"
            ;;
        other_bkgs|other-bkgs|other)
            era_datasets="$(
                strip_trailing_comma \
                    "${bkg_vv}${bkg_vvv}${bkg_vh_htt}${bkg_higgs}"
            )"
            ;;
        signal)
            era_datasets="$selected_signal_all"
            ;;
        ggphi|ggf)
            era_datasets="$selected_signal_ggphi"
            ;;
        bbphi|bbh)
            era_datasets="$selected_signal_bbphi"
            ;;
        datacard)
            era_data="$(
                data_for_config \
                    "$cfg"
            )"
            era_bkgs="$(
                strip_trailing_comma \
                    "$bkgs"
            )"
            era_datasets="${era_data},${era_bkgs},${datacard_signal_datasets}"
            ;;
    esac
    if [[ -n "$datasets_group" ]]; then
        datasets_group="${datasets_group}:"
    fi
    datasets_group="${datasets_group}${era_datasets}"
done
# Keep nominal QCD plots on PlotVariables1D with statistical errors enabled.
requested_hooks="$(get_extra_option_value --hist-hooks || true)"
if [[ "$stage" == "plot1d" && ",${requested_hooks}," == *,qcd,* ]]; then
    if ! has_extra_option --hide-stat-errors; then
        extra_args+=(--hide-stat-errors False)
    fi
fi
# The combined background plot includes data; production groups stay disjoint.
if [[ "$stage" == "plot1d" || "$stage" == "plot1d-shifted" ]]; then
    case "$sample_group" in
        backgrounds|background|bkg)
            plot_processes="$processes_bkg"
            datasets_group=""
            for cfg in "${config_array[@]}"; do
                era_datasets="$(data_for_config "$cfg"),$(strip_trailing_comma "$bkgs")"
                [[ -z "$datasets_group" ]] || datasets_group+=":"
                datasets_group+="$era_datasets"
            done
            ;;
    esac
fi
requested_version="$(get_extra_option_value --version || true)"
if [[ "$version" != "all_mass_bdt_compact_v1" || "${requested_version:-$version}" != "$version" ]]; then
    echo "ERROR: this submission script uses only all_mass_bdt_compact_v1." >&2
    exit 1
fi
# =============================================================================
# Shift definitions
# =============================================================================
kinematic_shifts="nominal,jec_*,jer_*"
hist_shifts="${kinematic_shifts},unclustered_*,recoilresp_*,recoilres_*"
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
shift_sources=$(
    IFS=,
    echo "${shift_sources_list[*]}"
)
# Data is nominal only.
if [[ "$sample_group" == "data" ]]; then
    kinematic_shifts="nominal"
    hist_shifts="nominal"
fi
# Default downstream production is nominal. Explicit --shifts still overrides it.
case "$stage" in
    produce|create-hists|merge-hists|check-hist-inputs)
        if ! has_extra_option --shifts; then
            hist_shifts="nominal"
        fi
        ;;
esac
if [[ "$stage" == "plot1d" ]] && ! has_extra_option --cf.PlotVariables1D-workflow; then
    extra_args+=(--cf.PlotVariables1D-workflow "$default_workflow")
fi
if [[ "$stage" == "plot1d-shifted" ]] && ! has_extra_option --cf.PlotShiftedVariables1D-workflow; then
    extra_args+=(--cf.PlotShiftedVariables1D-workflow "$default_workflow")
fi
# =============================================================================
# Workflow arguments
# =============================================================================
workflow_args=(
    --cf.CalibrateEvents-workflow
    "$calibrate_workflow"
    --cf.SelectEvents-workflow
    "$select_workflow"
    --cf.ReduceEvents-workflow
    "$reduce_workflow"
    --cf.MergeReducedEvents-workflow
    "$merge_reduced_workflow"
    --cf.ProduceColumns-workflow
    "$produce_workflow"
    --cf.CreateHistograms-workflow
    "$create_hists_workflow"
    --cf.MergeHistograms-workflow
    "$merge_hists_workflow"
    --cf.MergeShiftedHistograms-workflow
    "$merge_shifted_workflow"
)
# =============================================================================
# HTCondor parallel job settings
# =============================================================================
parallel_args=()
add_parallel_arg() {
    local option="$1"
    if ! has_extra_option "$option"; then
        parallel_args+=(
            "$option"
            "$parallel_jobs"
        )
    fi
}
if [[ "$calibrate_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.CalibrateEvents-parallel-jobs
fi
if [[ "$select_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.SelectEvents-parallel-jobs
fi
if [[ "$reduce_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.ReduceEvents-parallel-jobs
fi
if [[ "$merge_reduced_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.MergeReducedEvents-parallel-jobs
fi
if [[ "$produce_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.ProduceColumns-parallel-jobs
fi
if [[ "$create_hists_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.CreateHistograms-parallel-jobs
fi
if [[ "$merge_hists_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.MergeHistograms-parallel-jobs
fi
if [[ "$merge_shifted_workflow" == "htcondor" ]]; then
    add_parallel_arg \
        --cf.MergeShiftedHistograms-parallel-jobs
fi
# =============================================================================
# Common arguments
# =============================================================================
common_args=(
    --version "$version"
    --configs "$config"
    --pilot "$pilot"
    # Always consume MergeReducedEvents downstream instead of falling
    # back to individual ReduceEvents outputs when the second-stage
    # merging factor happens to be 1.
    --cf.ProvideReducedEvents-force-merging True
    "${workflow_args[@]}"
    "${parallel_args[@]}"
)
# Allow callers such as MSSM_submit_bdt_campaign.sh to explicitly restrict
# the dataset list. This is required for signal mass-block filtering.
if ! has_extra_option "--datasets"; then
    common_args+=(--datasets "$datasets_group")
fi
# =============================================================================
# Plot argument builder
# =============================================================================
prepare_plot_args() {
    plot_args=(
        --version
        "$version"
        --configs
        "$config"
        --datasets
        "$datasets_group"
        --pilot "$pilot"
        --cf.ProvideReducedEvents-force-merging
        True
        --calibrators
        main
        --selector
        main
        "${workflow_args[@]}"
        "${parallel_args[@]}"
    )
    # -------------------------------------------------------------------------
    # Processes
    # -------------------------------------------------------------------------
    if ! has_extra_option \
        --processes
    then
        plot_args+=(
            --processes
            "$plot_processes"
        )
    fi
    # -------------------------------------------------------------------------
    # Datacard mode
    # -------------------------------------------------------------------------
    if [[ "$sample_group" == "datacard" ]]; then
        if ! has_extra_option \
            --variables
        then
            plot_args+=(
                --variables
                "$datacard_variable"
            )
        fi
        if ! has_extra_option \
            --categories
        then
            plot_args+=(
                --categories
                "$datacard_category"
            )
        fi
        if ! has_extra_option \
            --producers
        then
            plot_args+=(
                --producers
                "$datacard_producers"
            )
        fi
        if ! has_extra_option \
            --hist-hooks
        then
            plot_args+=(
                --hist-hooks
                qcd
            )
        fi
    else
        if ! has_extra_option \
            --variables
        then
            plot_args+=(
                --variables
                "$variables"
            )
        fi
        if [[ -n "${categories:-}" ]] && \
           ! has_extra_option \
               --categories
        then
            plot_args+=(
                --categories
                "$categories"
            )
        fi
        if ! has_extra_option \
            --producers
        then
            plot_args+=(
                --producers
                main
            )
        fi
    fi
    # -------------------------------------------------------------------------
    # Output format
    # -------------------------------------------------------------------------
    if ! has_extra_option \
        --file-types
    then
        plot_args+=(
            --file-types
            png
        )
    fi
    # -------------------------------------------------------------------------
    # Plot settings
    # -------------------------------------------------------------------------
    if ! has_extra_option \
        --general-settings
    then
        plot_args+=(
            --general-settings
            "cms-label=pw"
        )
    fi
}
# =============================================================================
# Command execution
# =============================================================================
check_existing_inputs() {
    local check_stage="${1:-merge-hists}"
    [[ "$check_stage" != "plot1d" ]] || check_stage="merge-hists"
    local check_datasets check_variables check_shifts check_hist_producer
    local check_shift_sources
    local check_producers check_calibrators check_selector check_index check_dataset_index
    local check_dataset_groups check_args
    check_datasets="$(get_extra_option_value --datasets || true)"
    check_datasets="${check_datasets:-$datasets_group}"
    check_variables="$(get_extra_option_value --variables || true)"
    check_variables="${check_variables:-$variables}"
    check_shifts="$(get_extra_option_value --shifts || get_extra_option_value --shift || true)"
    if [[ -z "$check_shifts" ]]; then
        if [[ "$stage" == "plot1d" ]]; then
            check_shifts="nominal"
        else
            check_shifts="$hist_shifts"
        fi
    fi
    check_hist_producer="$(get_extra_option_value --hist-producer || true)"
    check_shift_sources="$(get_extra_option_value --shift-sources || true)"
    check_producers="$(get_extra_option_value --producers || true)"
    check_calibrators="$(get_extra_option_value --calibrators || true)"
    check_selector="$(get_extra_option_value --selector || true)"
    IFS=':' read -r -a check_dataset_groups <<< "$check_datasets"
    if (( ${#check_dataset_groups[@]} != 1 && ${#check_dataset_groups[@]} != ${#config_array[@]} )); then
        echo "ERROR: --datasets needs one CSV list or one colon-separated list per config." >&2
        return 1
    fi
    for ((check_index = 0; check_index < ${#config_array[@]}; check_index++)); do
        check_dataset_index=0
        if (( ${#check_dataset_groups[@]} > 1 )); then
            check_dataset_index="$check_index"
        fi
        check_args=(
            --config "${config_array[$check_index]}"
            --datasets "${check_dataset_groups[$check_dataset_index]}"
            --stage "$check_stage" --variables "$check_variables" --shifts "$check_shifts"
            --producers "${check_producers:-main}" --calibrators "${check_calibrators:-main}"
            --selector "${check_selector:-main}" --version "$version"
        )
        if [[ -n "$check_hist_producer" ]]; then
            check_args+=(--hist-producer "$check_hist_producer")
        fi
        if [[ -n "$histogram_bdt_masses" ]]; then
            check_args+=(--histogram-bdt-masses "$histogram_bdt_masses")
        fi
        if [[ "$check_stage" == "merge-shifted" ]]; then
            check_args+=(--shift-sources "${check_shift_sources:-$shift_sources}")
        fi
        python "${SCRIPT_DIR}/MSSM_check_histogram_inputs.py" "${check_args[@]}" || return "$?"
    done
}
run_command() {
    if [[ -n "$histogram_bdt_masses" ]]; then
        set -- "$@" --histogram-bdt-masses "$histogram_bdt_masses"
    fi
    case "$stage" in
        produce|create-hists|merge-hists|plot1d)
            check_existing_inputs "$stage" || return "$?"
            ;;
        merge-shifted|plot1d-shifted)
            # Check both external shape shifts and embedded weight shifts through
            # the actual MergeShiftedHistograms dependencies, before scheduling.
            check_existing_inputs merge-shifted || return "$?"
            ;;
    esac
    echo
    echo "================================================================================"
    echo "Stage             : $stage"
    echo "Config option     : $config_option"
    echo "Configs           : $config"
    echo "Sample group      : $sample_group"
    if [[ "$stage" == "plot1d" || "$stage" == "plot1d-shifted" ]]; then
        echo "Processes         : $plot_processes"
    fi
    if (( ${#requested_bdt_masses[@]} > 0 )); then
        local IFS=,
        echo "BDT masses        : ${requested_bdt_masses[*]}"
    fi
    echo "Parallel jobs     : $parallel_jobs"
    echo
    echo "Workflows:"
    echo "  CalibrateEvents        : $calibrate_workflow"
    echo "  SelectEvents           : $select_workflow"
    echo "  ReduceEvents           : $reduce_workflow"
    echo "  MergeReducedEvents     : $merge_reduced_workflow"
    echo "  ProduceColumns         : $produce_workflow"
    echo "  CreateHistograms       : $create_hists_workflow"
    echo "  MergeHistograms        : $merge_hists_workflow"
    echo "  MergeShiftedHistograms : $merge_shifted_workflow"
    echo "================================================================================"
    echo
    printf 'law run'
    printf ' %s' "$@"
    echo
    echo
    PYTHONFAULTHANDLER=1 \
        law run "$@"
}

launch_signal_masses() {
    local mass_text mass vars dataset_name arg child_index failed=0
    local next_mass=0 finished_pid="" status=0 pending_pid
    local script_path="${SCRIPT_DIR}/${BASH_SOURCE[0]##*/}"
    local -a masses=() child_args=() mass_args=()
    local -A active_masses=()

    case "$sample_group" in
        ggphi|ggf|bbphi|bbh|signal) ;;
        *)
            echo "ERROR: --signal-masses requires ggphi, bbphi or signal." >&2
            return 1
            ;;
    esac

    case "$stage" in
        produce|create-hists|merge-hists|merge-shifted|check-hist-inputs) ;;
        *)
            echo "ERROR: --signal-masses applies to production and histogram stages." >&2
            return 1
            ;;
    esac

    if has_extra_option --variables || has_extra_option --datasets || [[ -n "$histogram_bdt_masses" ]]; then
        echo "ERROR: --signal-masses automatically supplies matching BDT variables and datasets." >&2
        echo "Use --signal-masses without --variables, --datasets or --histogram-bdt-masses." >&2
        return 1
    fi

    # -------------------------------------------------------------------------
    # Resolve requested masses
    # -------------------------------------------------------------------------
    mass_text="$(python - "$signal_masses" <<'PY'
import sys
from MSSM_H_tt.config.mass_points import read_bdt_masses

grid = read_bdt_masses()
choice = sys.argv[1].strip()

try:
    selected = grid if choice == "all" else list(dict.fromkeys(int(m.strip()) for m in choice.split(",")))
except ValueError:
    raise SystemExit("ERROR: --signal-masses requires all or integer mass points.")

if not selected or any(m not in grid for m in selected):
    raise SystemExit("ERROR: --signal-masses contains an unconfigured mass point.")

print(*selected, sep="\n")
PY
    )" || return 1

    mapfile -t masses <<< "$mass_text"

    for mass in "${masses[@]}"; do
        [[ "$mass" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: invalid configured signal mass '$mass'." >&2
            return 1
        }
    done

    # -------------------------------------------------------------------------
    # Prepare child arguments
    # -------------------------------------------------------------------------
    for ((child_index = 0; child_index < ${#raw_extra_args[@]}; child_index++)); do
        arg="${raw_extra_args[$child_index]}"

        case "$arg" in
            --signal-masses|--parallel-masses)
                ((child_index += 1))
                ;;
            --signal-masses=*|--parallel-masses=*)
                ;;
            *)
                child_args+=("$arg")
                ;;
        esac
    done

    if ! has_extra_option --workers; then
        child_args+=(--workers 1)
    fi

    case "$stage" in
        produce|create-hists|merge-hists|check-hist-inputs)
            if ! has_extra_option --shifts; then
                child_args+=(--shifts 'nominal,jec_*,jer_*,unclustered_*,recoilresp_*,recoilres_*')
            fi
            ;;
    esac

    # -------------------------------------------------------------------------
    # Launch signal masses with limited concurrency
    # -------------------------------------------------------------------------
    echo
    echo "================================================================================"
    echo "Signal mass processing"
    echo "Stage                : $stage"
    echo "Config               : $config_option"
    echo "Sample group         : $sample_group"
    echo "Total masses         : ${#masses[@]}"
    echo "Parallel masses      : $parallel_masses"
    echo "Workers per mass     : $(get_extra_option_value --workers || echo 1)"
    echo "================================================================================"
    echo

    # Refill available slots whenever a mass finishes.
    while (( next_mass < ${#masses[@]} || ${#active_masses[@]} > 0 )); do

        # Launch new masses until the concurrency limit is reached.
        while (( next_mass < ${#masses[@]} && ${#active_masses[@]} < parallel_masses )); do

            mass="${masses[$next_mass]}"

            case "$sample_group" in
                ggphi|ggf)
                    dataset_name="ggphi_phitt_${mass}"
                    ;;
                bbphi|bbh)
                    dataset_name="bbphi_phitt_${mass}"
                    ;;
                signal)
                    dataset_name="ggphi_phitt_${mass},bbphi_phitt_${mass}"
                    ;;
            esac

            vars="bdt_D_sig_vs_Disc_ggphi_M${mass},bdt_D_sig_vs_Disc_bbphi_M${mass},bdt_D_DY_M${mass},bdt_D_TT_M${mass}"

            mass_args=(--datasets "$dataset_name")

            if [[ "$stage" != "produce" ]]; then
                mass_args+=(--variables "$vars")
            fi

            echo "[M${mass}] Starting $stage"

            # Stream output to the terminal without creating extra log files.
            PYTHONUNBUFFERED=1 bash "$script_path" \
                "$config_option" "$sample_group" "$stage" \
                "${child_args[@]}" "${mass_args[@]}" 2>&1 \
                | sed -u "s/^/[M${mass}] /" &

            active_masses["$!"]="$mass"

            ((next_mass += 1))
        done

        # ---------------------------------------------------------------------
        # Wait for any active mass to finish
        # ---------------------------------------------------------------------
        finished_pid=""

        if wait -n -p finished_pid "${!active_masses[@]}"; then
            status=0
        else
            status=$?
        fi

        if [[ -z "$finished_pid" ]]; then
            echo "ERROR: could not identify the completed mass process." >&2

            for pending_pid in "${!active_masses[@]}"; do
                if ! wait "$pending_pid"; then
                    echo "ERROR: M${active_masses[$pending_pid]} failed." >&2
                fi
            done

            failed=1
            break
        fi

        # ---------------------------------------------------------------------
        # Check completion status
        # ---------------------------------------------------------------------
        mass="${active_masses[$finished_pid]}"

        if (( status != 0 )); then
            echo "[M${mass}] ERROR: $stage failed (exit status $status)." >&2
            failed=1
        else
            echo "[M${mass}] Finished successfully."
        fi

        unset "active_masses[$finished_pid]"

    done

    echo
    echo "================================================================================"

    if (( failed == 0 )); then
        echo "All signal masses completed successfully."
    else
        echo "Signal processing finished with errors."
    fi

    echo "================================================================================"

    return "$failed"
}

if [[ -n "$signal_masses" ]]; then
    launch_signal_masses
    exit "$?"
fi
# =============================================================================
# Stage dispatch
# =============================================================================
case "$stage" in
    # Read-only check: refuse missing upstream event outputs before submission.
    check-hist-inputs)
        check_existing_inputs merge-hists
        ;;
    # -------------------------------------------------------------------------
    # CalibrateEvents
    # -------------------------------------------------------------------------
    calibrate)
        run_command \
            cf.CalibrateEventsWrapper \
            "${common_args[@]}" \
            --calibrator main \
            --shifts "$kinematic_shifts" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # SelectEvents
    # -------------------------------------------------------------------------
    select)
        run_command \
            cf.SelectEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # ReduceEvents
    # -------------------------------------------------------------------------
    reduce)
        run_command \
            cf.ReduceEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # MergeReducedEvents
    # -------------------------------------------------------------------------
    merge-reduced)
        run_command \
            cf.MergeReducedEventsWrapper \
            "${common_args[@]}" \
            --calibrators main \
            --selector main \
            --shifts "$kinematic_shifts" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # ProduceColumns
    # -------------------------------------------------------------------------
    produce)
        produce_args=(
            "${common_args[@]}"
            --shifts
            "$hist_shifts"
        )
        if ! has_extra_option \
            --producers
        then
            produce_args+=(
                --producers
                main
            )
        fi
        run_command \
            cf.ProduceColumnsWrapper \
            "${produce_args[@]}" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # CreateHistograms
    # -------------------------------------------------------------------------
    create-hists)
        create_hists_args=(
            "${common_args[@]}"
            --calibrators
            main
            --selector
            main
            --shifts
            "$hist_shifts"
        )
        if ! has_extra_option \
            --producers
        then
            create_hists_args+=(
                --producers
                main
            )
        fi
        if ! has_extra_option \
            --variables
        then
            create_hists_args+=(
                --variables
                "$variables"
            )
        fi
        run_command \
            cf.CreateHistogramsWrapper \
            "${create_hists_args[@]}" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # MergeHistograms
    # -------------------------------------------------------------------------
    merge-hists)
        merge_hists_args=(
            "${common_args[@]}"
            --calibrators
            main
            --selector
            main
            --shifts
            "$hist_shifts"
        )
        if ! has_extra_option \
            --producers
        then
            merge_hists_args+=(
                --producers
                main
            )
        fi
        if ! has_extra_option \
            --variables
        then
            merge_hists_args+=(
                --variables
                "$variables"
            )
        fi
        run_command \
            cf.MergeHistogramsWrapper \
            "${merge_hists_args[@]}" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # MergeShiftedHistograms
    # -------------------------------------------------------------------------
    merge-shifted)
        if [[ "$sample_group" == "data" ]]; then
            echo "Data has no shifted histogram merge."
            echo "Use merge-hists for nominal data histograms."
            exit 0
        fi
        merge_shifted_args=(
            "${common_args[@]}"
            --calibrators
            main
            --selector
            main
        )
        if ! has_extra_option \
            --producers
        then
            merge_shifted_args+=(
                --producers
                main
            )
        fi
        if ! has_extra_option \
            --variables
        then
            merge_shifted_args+=(
                --variables
                "$variables"
            )
        fi
        if ! has_extra_option \
            --shift-sources
        then
            merge_shifted_args+=(
                --shift-sources
                "$shift_sources"
            )
        fi
        run_command \
            cf.MergeShiftedHistogramsWrapper \
            "${merge_shifted_args[@]}" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # PlotVariables1D
    # -------------------------------------------------------------------------
    plot1d)
        prepare_plot_args
        run_command \
            cf.PlotVariables1D \
            "${plot_args[@]}" \
            --shift nominal \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # PlotShiftedVariables1D
    # -------------------------------------------------------------------------
    plot1d-shifted)
        prepare_plot_args
        if ! has_extra_option \
            --shift-sources
        then
            plot_args+=(
                --shift-sources
                "$shift_sources"
            )
        fi
        run_command \
            cf.PlotShiftedVariables1D \
            "${plot_args[@]}" \
            "${extra_args[@]}"
        ;;
    # -------------------------------------------------------------------------
    # Unknown stage
    # -------------------------------------------------------------------------
    *)
        echo "ERROR: unknown stage '$stage'." >&2
        exit 1
        ;;
esac