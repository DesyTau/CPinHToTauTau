#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# Usage
# =============================================================================

if [[ $# -ne 2 ]]; then
    echo "Usage:"
    echo "  $0 <configuration> <stage>"
    echo
    echo "Configurations:"
    echo "  22_emu"
    echo "  22EE_emu"
    echo "  23_emu"
    echo "  23BPix_emu"
    echo "  22and23_emu"
    echo
    echo "Stages:"
    echo "  calibrate"
    echo "  select"
    echo "  selection-stats"
    echo "  reduce"
    echo "  reduction-stats"
    echo "  merge-reduced"
    echo "  produce"
    echo "  create-histograms"
    echo "  merge-histograms"
    echo "  merge-shifted-histograms"
    exit 1
fi

configuration_option="$1"
stage="$2"


# =============================================================================
# Setup
# =============================================================================

script_dir="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${script_dir}/common_run3_MSSM.sh"

set_common_vars "$configuration_option"


# =============================================================================
# Local submission settings
#
# Only one Luigi worker and one submission thread are used on lxplus.
#
# parallel_jobs=0 means unlimited on the LAW remote workflow side, i.e. submit
# all branches of the current dataset during this invocation.
#
# Since we process only ONE dataset per law invocation, this does not create
# the huge Luigi graph that caused the segfault.
# =============================================================================

workers="${WORKERS:-1}"
submission_threads="${SUBMISSION_THREADS:-1}"
parallel_jobs="${PARALLEL_JOBS:-0}"


# =============================================================================
# Histogram variables
# =============================================================================

first_bdt_mass="$(
python - <<'PY'
from MSSM_H_tt.config.mass_points import read_bdt_masses

print(read_bdt_masses()[0])
PY
)"

bdt_variable="bdt_D_DY_M${first_bdt_mass}"
inclusive_variable="emu_mt_tot"


# =============================================================================
# Validate stage
# =============================================================================

case "$stage" in
    calibrate|\
    select|\
    selection-stats|\
    reduce|\
    reduction-stats|\
    merge-reduced|\
    produce|\
    create-histograms|\
    merge-histograms|\
    merge-shifted-histograms)
        ;;
    *)
        echo "ERROR: unknown stage '$stage'"
        exit 1
        ;;
esac


# =============================================================================
# Config list
#
# For 22and23_emu, $config contains several comma-separated configurations.
# =============================================================================

IFS=',' read -r -a configs <<< "$config"


echo
echo "================================================================"
echo "MSSM no-poll submission"
echo "================================================================"
echo "Configuration option : $configuration_option"
echo "Configs              : $config"
echo "Stage                : $stage"
echo "Version              : $version"
echo "Workers              : $workers"
echo "Submission threads   : $submission_threads"
echo "Parallel jobs        : $parallel_jobs"
echo "BDT seed variable    : $bdt_variable"
echo "Inclusive variable   : $inclusive_variable"
echo "CF_JOB_BASE          : ${CF_JOB_BASE:-<not set>}"
echo "================================================================"
echo


# =============================================================================
# Loop over configurations
# =============================================================================

for config_name in "${configs[@]}"; do

    echo
    echo "################################################################"
    echo "# Config: $config_name"
    echo "################################################################"
    echo


    # =========================================================================
    # Read the dataset list and shifts directly from the config.
    #
    # Output format:
    #
    #   KINEMATIC <tab> shift1,shift2,...
    #   SOURCES   <tab> source1,source2,...
    #   DATASET   <tab> dataset_name <tab> 0/1
    #
    # The last field is 1 for data and 0 for MC.
    #
    # data_tau_* is deliberately excluded.
    # =========================================================================

    metadata="$(
        python - "$config_name" <<'PY'
import re
import sys

from MSSM_H_tt.config.analysis_MSSM_H_tt_skim_2025_v1 import (
    analysis_MSSM_H_tt_skim_2025_v1 as analysis,
)


config_name = sys.argv[1]
config_inst = analysis.get_config(config_name)


# -------------------------------------------------------------------------
# Kinematic shifts
#
# bdt_input currently identifies:
#   - unclustered MET
#   - JEC
#   - JER
#   - recoil response/resolution
# -------------------------------------------------------------------------

kinematic_shifts = ["nominal"]

for shift_inst in config_inst.shifts:
    if (
        shift_inst.name != "nominal"
        and shift_inst.has_tag("bdt_input")
    ):
        kinematic_shifts.append(
            shift_inst.name
        )

print(
    "KINEMATIC\t"
    + ",".join(kinematic_shifts)
)


# -------------------------------------------------------------------------
# Sources needed by MergeShiftedHistograms
#
# Weight-only sources are embedded into the nominal histogram in the current
# histogramming implementation.
#
# Still pass them as shift sources here. MergeShiftedHistograms knows which
# ones are embedded and therefore does not require separate MergeHistograms
# tasks for them.
# -------------------------------------------------------------------------

weight_sources = set(
    config_inst.x.histogram_weight_shift_sources
)

# Keep a stable, readable ordering.
weight_source_order = (
    "muon_weight",
    "electron_weight",
    "Trigger_SF_weight",
    "pu_weight",
    "top_pt_weight",
    "zpt_weight",

    "CMS_PS_ISR",
    "CMS_PS_FSR",
    "CMS_Scale_muR",
    "CMS_Scale_muF",

    "btag_weight_hf",
    "btag_weight_lf",
    "btag_weight_hfstats1",
    "btag_weight_hfstats2",
    "btag_weight_lfstats1",
    "btag_weight_lfstats2",
    "btag_weight_cferr1",
    "btag_weight_cferr2",
)

shift_sources = []

for source in weight_source_order:
    if source in weight_sources:
        shift_sources.append(source)

# Include any future embedded source that is not in the explicit ordering.
for source in sorted(
    weight_sources - set(shift_sources)
):
    shift_sources.append(source)


def shift_to_source(shift_name):
    return re.sub(
        r"_(up|down)$",
        "",
        shift_name,
    )


# Add the genuine kinematic sources.
for shift_inst in config_inst.shifts:
    if not shift_inst.has_tag("bdt_input"):
        continue

    source = shift_to_source(
        shift_inst.name
    )

    if source not in shift_sources:
        shift_sources.append(source)


print(
    "SOURCES\t"
    + ",".join(shift_sources)
)


# -------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------

for dataset_inst in sorted(
    config_inst.datasets,
    key=lambda dataset: dataset.name,
):
    if dataset_inst.name.startswith(
        "data_tau_"
    ):
        continue

    print(
        "DATASET\t"
        f"{dataset_inst.name}\t"
        f"{int(dataset_inst.is_data)}"
    )
PY
    )"


    # =========================================================================
    # Parse metadata
    # =========================================================================

    kinematic_shifts_csv="$(
        printf '%s\n' "$metadata" |
        awk -F $'\t' '$1 == "KINEMATIC" {print $2; exit}'
    )"

    shift_sources_csv="$(
        printf '%s\n' "$metadata" |
        awk -F $'\t' '$1 == "SOURCES" {print $2; exit}'
    )"

    mapfile -t dataset_lines < <(
        printf '%s\n' "$metadata" |
        awk -F $'\t' '$1 == "DATASET" {print $2 "\t" $3}'
    )


    if [[ -z "$kinematic_shifts_csv" ]]; then
        echo "ERROR: could not determine kinematic shifts for $config_name"
        exit 1
    fi

    if [[ ${#dataset_lines[@]} -eq 0 ]]; then
        echo "ERROR: no datasets found for $config_name"
        exit 1
    fi


    echo "Kinematic shifts:"
    echo "  $kinematic_shifts_csv"
    echo
    echo "Histogram shift sources:"
    echo "  $shift_sources_csv"
    echo
    echo "Datasets:"
    echo "  ${#dataset_lines[@]}"
    echo


    # =========================================================================
    # Loop over datasets
    # =========================================================================

    for i in "${!dataset_lines[@]}"; do

        IFS=$'\t' read -r dataset is_data <<< "${dataset_lines[$i]}"


        # Data only needs nominal.
        if [[ "$is_data" == "1" ]]; then
            dataset_shifts="nominal"
            dataset_type="data"
        else
            dataset_shifts="$kinematic_shifts_csv"
            dataset_type="MC"
        fi


        echo
        echo "================================================================"
        echo "Dataset $((i + 1))/${#dataset_lines[@]}"
        echo "  config  : $config_name"
        echo "  dataset : $dataset"
        echo "  type    : $dataset_type"
        echo "  stage   : $stage"
        echo "================================================================"
        echo


        # =====================================================================
        # Common remote arguments
        # =====================================================================

        common_args=(
            --configs "$config_name"
            --datasets "$dataset"
            --version "$version"

            --workflow "htcondor"

            --workers "$workers"

            --no-poll "True"
            --submission-threads "$submission_threads"

            --parallel-jobs "$parallel_jobs"

            --pilot "True"
        )


        # =====================================================================
        # Stage
        # =====================================================================

        case "$stage" in

            # -----------------------------------------------------------------
            # Calibration
            # -----------------------------------------------------------------

            calibrate)

                law run cf.CalibrateEventsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --calibrator "main"
                ;;


            # -----------------------------------------------------------------
            # Selection
            # -----------------------------------------------------------------

            select)

                law run cf.SelectEventsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts"
                ;;


            # -----------------------------------------------------------------
            # Merge selection statistics
            # -----------------------------------------------------------------

            selection-stats)

                law run cf.MergeSelectionStatsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts"
                ;;


            # -----------------------------------------------------------------
            # Reduction
            # -----------------------------------------------------------------

            reduce)

                law run cf.ReduceEventsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts"
                ;;


            # -----------------------------------------------------------------
            # Determine reduction merging factors
            # -----------------------------------------------------------------

            reduction-stats)

                law run cf.MergeReductionStatsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts"
                ;;


            # -----------------------------------------------------------------
            # Merge reduced events
            # -----------------------------------------------------------------

            merge-reduced)

                law run cf.MergeReducedEventsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts"
                ;;


            # -----------------------------------------------------------------
            # Produce columns
            # -----------------------------------------------------------------

            produce)

                law run cf.ProduceColumnsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --producers "main"
                ;;


            # -----------------------------------------------------------------
            # Create histograms
            #
            # Keep BDT and inclusive variables as separate tasks so their task
            # identities agree with the final plotting workflow.
            # -----------------------------------------------------------------

            create-histograms)

                echo
                echo "--- BDT histograms ---"
                echo

                law run cf.CreateHistogramsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --variables "$bdt_variable"

                echo
                echo "--- Inclusive histogram ---"
                echo

                law run cf.CreateHistogramsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --variables "$inclusive_variable"
                ;;


            # -----------------------------------------------------------------
            # Merge histograms for each kinematic shift
            # -----------------------------------------------------------------

            merge-histograms)

                echo
                echo "--- BDT histograms ---"
                echo

                law run cf.MergeHistogramsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --variables "$bdt_variable"

                echo
                echo "--- Inclusive histogram ---"
                echo

                law run cf.MergeHistogramsWrapper \
                    "${common_args[@]}" \
                    --shifts "$dataset_shifts" \
                    --variables "$inclusive_variable"
                ;;


            # -----------------------------------------------------------------
            # Merge shifted histograms
            #
            # Data stops at nominal MergeHistograms.
            # -----------------------------------------------------------------

            merge-shifted-histograms)

                if [[ "$is_data" == "1" ]]; then
                    echo "Data dataset: MergeShiftedHistograms not required."
                    continue
                fi

                echo
                echo "--- BDT histograms ---"
                echo

                law run cf.MergeShiftedHistogramsWrapper \
                    "${common_args[@]}" \
                    --shift-sources "$shift_sources_csv" \
                    --variables "$bdt_variable"

                echo
                echo "--- Inclusive histogram ---"
                echo

                law run cf.MergeShiftedHistogramsWrapper \
                    "${common_args[@]}" \
                    --shift-sources "$shift_sources_csv" \
                    --variables "$inclusive_variable"
                ;;

        esac

    done

done


echo
echo "================================================================"
echo "Submission pass completed"
echo "  configuration : $configuration_option"
echo "  stage         : $stage"
echo "================================================================"