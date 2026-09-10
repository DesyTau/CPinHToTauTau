#!/bin/bash

source ./common_run3_MSSM.sh


# -------------------------------------------------------------------------
# Common configuration
# -------------------------------------------------------------------------

if [[ -z "$1" ]]; then
    echo "ERROR: no configuration option provided"
    echo
    echo "Usage:"
    echo "  $0 <configuration> --workers <N> --workflow <workflow> [extra law arguments]"
    echo
    echo "Examples:"
    echo "  $0 22_emu --workers 10 --workflow local"
    echo "  $0 22and23_emu --workers 20 --workflow htcondor"
    echo
    echo "Available options:"
    echo "  22and23_emu"
    echo "  22_emu"
    echo "  22EE_emu"
    echo "  23_emu"
    echo "  23BPix_emu"
    exit 1
fi

configuration_option="$1"

if ! set_common_vars "$configuration_option"; then
    exit 1
fi


# ============================================================================
# Parse command-line arguments
#
# --workers and --workflow are extracted and passed to ALL tasks.
#
# Other extra arguments are passed only to ReduceEvents, so HTCondor-specific
# arguments cannot accidentally be forwarded to MergeSelectionMasks.
# ============================================================================

workers=""
workflow=""
extra_args=()

shift

while [[ $# -gt 0 ]]; do

    case "$1" in

        --workers)
            if [[ -z "$2" ]]; then
                echo "ERROR: --workers requires a value"
                exit 1
            fi

            workers="$2"
            shift 2
            ;;

        --workers=*)
            workers="${1#*=}"
            shift
            ;;

        --workflow)
            if [[ -z "$2" ]]; then
                echo "ERROR: --workflow requires a value"
                exit 1
            fi

            workflow="$2"
            shift 2
            ;;

        --workflow=*)
            workflow="${1#*=}"
            shift
            ;;

        *)
            extra_args+=("$1")
            shift
            ;;

    esac

done


# -------------------------------------------------------------------------
# Validate workers
# -------------------------------------------------------------------------

if [[ -z "$workers" ]]; then
    echo "ERROR: --workers was not provided"
    echo
    echo "Example:"
    echo "  $0 22_emu --workers 10 --workflow htcondor"
    exit 1
fi

if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --workers must be a positive integer"
    echo "Received: $workers"
    exit 1
fi


# -------------------------------------------------------------------------
# Validate workflow
# -------------------------------------------------------------------------

if [[ -z "$workflow" ]]; then
    echo "ERROR: --workflow was not provided"
    echo
    echo "Available workflows:"
    echo "  local"
    echo "  htcondor"
    echo
    echo "Example:"
    echo "  $0 22_emu --workers 10 --workflow htcondor"
    exit 1
fi

case "$workflow" in
    local|htcondor)
        ;;
    *)
        echo "ERROR: unsupported workflow: $workflow"
        echo
        echo "Available workflows:"
        echo "  local"
        echo "  htcondor"
        exit 1
        ;;
esac


# -------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------

if [[ -z "$config" ]]; then
    echo "ERROR: config is empty"
    exit 1
fi

if [[ -z "$datasets" ]]; then
    echo "ERROR: datasets is empty"
    exit 1
fi

if [[ -z "$version" ]]; then
    echo "ERROR: version is empty"
    exit 1
fi


# ============================================================================
# JEC sources
# ============================================================================

# Sources common to all Run 3 eras
jec_sources=(
    jec_Regrouped_Absolute
    jec_Regrouped_BBEC1
    jec_Regrouped_EC2
    jec_Regrouped_HF
    jec_Regrouped_RelativeBal
    jec_Regrouped_FlavorQCD
)


# -------------------------------------------------------------------------
# Add era-dependent JEC sources for all requested configurations
# -------------------------------------------------------------------------

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
# Reduction shifts
#
# ReduceEvents must be produced for all shifts that modify the event
# selection:
#
#   - nominal
#   - all JEC variations
#   - JER up/down
# ============================================================================

reduction_shifts=(
    nominal
    jer_up
    jer_down
)

for source in "${jec_sources[@]}"; do
    reduction_shifts+=(
        "${source}_up"
        "${source}_down"
    )
done


# -------------------------------------------------------------------------
# Remove possible duplicates
# -------------------------------------------------------------------------

declare -A seen_shifts
unique_reduction_shifts=()

for shift_name in "${reduction_shifts[@]}"; do

    if [[ -z "${seen_shifts[$shift_name]}" ]]; then
        unique_reduction_shifts+=("$shift_name")
        seen_shifts[$shift_name]=1
    fi

done

reduction_shifts_csv="$(
    IFS=,
    echo "${unique_reduction_shifts[*]}"
)"


# -------------------------------------------------------------------------
# Final sanity check
# -------------------------------------------------------------------------

if [[ -z "$reduction_shifts_csv" ]]; then
    echo "ERROR: reduction_shifts_csv is empty"
    exit 1
fi


# ============================================================================
# Common arguments
#
# workers and workflow are identical for ALL tasks.
# ============================================================================

common_args=(
    --configs "$config"
    --datasets "$datasets"
    --version "$version"
    --calibrators "main"
    --selector "main"
    --shifts "$reduction_shifts_csv"
    --workers "$workers"
    --workflow "$workflow"
)


# ============================================================================
# ReduceEvents arguments
# ============================================================================

reduce_args=(
    "${common_args[@]}"
)


# -------------------------------------------------------------------------
# Add HTCondor-specific options only when using HTCondor
#
# --pilot is only passed to ReduceEvents because MergeSelectionMasks does
# not define that parameter.
# -------------------------------------------------------------------------

if [[ "$workflow" == "htcondor" ]]; then

    reduce_args+=(
        --poll-interval "5m"
        --pilot "True"
        --retries "10"
    )

fi


# -------------------------------------------------------------------------
# Other user-provided Law arguments are passed only to ReduceEvents
# -------------------------------------------------------------------------

reduce_args+=(
    "${extra_args[@]}"
)


# ============================================================================
# Merge arguments
#
# Same workers and same workflow as ReduceEvents.
#
# No --pilot is passed here.
# ============================================================================

merge_args=(
    "${common_args[@]}"
)


# ============================================================================
# Helper functions
# ============================================================================

run_reduce_step() {

    local task="$1"
    local description="$2"

    echo
    echo "======================================================================"
    echo "$description"
    echo "======================================================================"
    echo

    echo law run "$task" "${reduce_args[@]}"
    echo

    law run "$task" "${reduce_args[@]}"

    local status=$?

    if [[ $status -ne 0 ]]; then
        echo
        echo "ERROR: $description failed"
        echo
        echo "cf.SelectEvents will NOT be deleted."
        echo
        exit $status
    fi

    echo
    echo "$description completed successfully"
    echo
}


run_merge_step() {

    local task="$1"
    local description="$2"

    echo
    echo "======================================================================"
    echo "$description"
    echo "======================================================================"
    echo

    echo law run "$task" "${merge_args[@]}"
    echo

    law run "$task" "${merge_args[@]}"

    local status=$?

    if [[ $status -ne 0 ]]; then
        echo
        echo "ERROR: $description failed"
        echo
        echo "cf.SelectEvents will NOT be deleted."
        echo
        exit $status
    fi

    echo
    echo "$description completed successfully"
    echo
}


# ============================================================================
# Summary
# ============================================================================

echo
echo "======================================================================"
echo "ReduceEvents production"
echo "======================================================================"
echo "Configuration option: $configuration_option"
echo "Configs:              $config"
echo "Datasets:             $datasets"
echo "Version:              $version"
echo "Calibrators:          main"
echo "Selector:             main"
echo "Workers:              $workers"
echo "Workflow:             $workflow"
echo
echo "Execution:"
echo "  ReduceEvents:          $workflow"
echo "  MergeReducedEvents:    $workflow"
echo "  MergeSelectionStats:   $workflow"
echo "  MergeSelectionMasks:   $workflow"
echo
echo "Workers for every task: $workers"
echo
echo "Reduction shifts:"
for shift_name in "${unique_reduction_shifts[@]}"; do
    echo "  - $shift_name"
done
echo
echo "Number of reduction shifts: ${#unique_reduction_shifts[@]}"
echo
echo "After all merge steps succeed, the matching cf.SelectEvents"
echo "outputs for version '$version' will be removed."
echo
echo "======================================================================"
echo


# ============================================================================
# 1. ReduceEvents
# ============================================================================

run_reduce_step \
    cf.ReduceEventsWrapper \
    "ReduceEvents production"


# ============================================================================
# 2. MergeReducedEvents
# ============================================================================

run_merge_step \
    cf.MergeReducedEventsWrapper \
    "MergeReducedEvents production"


# ============================================================================
# 3. MergeSelectionStats
#
# MergeSelectionStats directly reads SelectEvents.
# Therefore SelectEvents must still exist at this point.
# ============================================================================

run_merge_step \
    cf.MergeSelectionStatsWrapper \
    "MergeSelectionStats production"


# ============================================================================
# 4. MergeSelectionMasks
#
# MergeSelectionMasks directly reads SelectEvents and produces the masks
# required later by the cutflow tasks.
#
# SelectEvents can only be removed after this task has completed.
# ============================================================================

run_merge_step \
    cf.MergeSelectionMasksWrapper \
    "MergeSelectionMasks production"


# ============================================================================
# 5. Delete corresponding cf.SelectEvents outputs
# ============================================================================

SELECT_EVENTS_BASE="/eos/project/d/desytau/public/jmalvaso/MSSM_H_tt_store/analysis_MSSM_H_tt_skim_2025_v1/cf.SelectEvents"


cleanup_select_events() {

    echo
    echo "======================================================================"
    echo "Cleaning cf.SelectEvents"
    echo "======================================================================"
    echo
    echo "Base:"
    echo "  $SELECT_EVENTS_BASE"
    echo
    echo "Version to remove:"
    echo "  $version"
    echo


    # ---------------------------------------------------------------------
    # Split configs
    # ---------------------------------------------------------------------

    local -a configs_array
    IFS=',' read -ra configs_array <<< "$config"


    # ---------------------------------------------------------------------
    # Split dataset groups
    #
    # For combined configurations:
    #
    #   datasets_cfg1:datasets_cfg2:datasets_cfg3:...
    # ---------------------------------------------------------------------

    local datasets_spec="$datasets"

    while [[ "$datasets_spec" == *: ]]; do
        datasets_spec="${datasets_spec%:}"
    done

    local -a dataset_groups
    IFS=':' read -ra dataset_groups <<< "$datasets_spec"


    # ---------------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------------

    if [[ ${#configs_array[@]} -ne ${#dataset_groups[@]} ]]; then

        echo "ERROR: number of configs and dataset groups does not match."
        echo
        echo "Configs:"
        printf '  - %s\n' "${configs_array[@]}"
        echo
        echo "Dataset groups:"
        printf '  - %s\n' "${dataset_groups[@]}"
        echo
        echo "No SelectEvents files were deleted."

        return 1

    fi


    local n_removed=0

    local i
    local cfg
    local dataset_group
    local dataset
    local dataset_root
    local version_dir
    local relative_path
    local matched_shift
    local shift_name


    # ---------------------------------------------------------------------
    # Loop over config -> dataset group
    # ---------------------------------------------------------------------

    for i in "${!configs_array[@]}"; do

        cfg="${configs_array[$i]}"
        dataset_group="${dataset_groups[$i]}"

        echo
        echo "------------------------------------------------------------------"
        echo "Config: $cfg"
        echo "------------------------------------------------------------------"

        local -a dataset_array
        IFS=',' read -ra dataset_array <<< "$dataset_group"


        for dataset in "${dataset_array[@]}"; do

            [[ -z "$dataset" ]] && continue

            dataset_root="${SELECT_EVENTS_BASE}/${cfg}/${dataset}"

            if [[ ! -d "$dataset_root" ]]; then

                echo "[skip] no SelectEvents directory for:"
                echo "       $cfg / $dataset"

                continue

            fi


            # -------------------------------------------------------------
            # Search only below this exact config/dataset.
            # -------------------------------------------------------------

            while IFS= read -r -d '' version_dir; do

                relative_path="${version_dir#${dataset_root}/}"
                matched_shift=""


                # ---------------------------------------------------------
                # Require one of the reduction shifts to occur as a full
                # path component.
                # ---------------------------------------------------------

                for shift_name in "${unique_reduction_shifts[@]}"; do

                    if [[ "/${relative_path}/" == *"/${shift_name}/"* ]]; then
                        matched_shift="$shift_name"
                        break
                    fi

                done


                # ---------------------------------------------------------
                # Do not remove outputs belonging to another shift.
                # ---------------------------------------------------------

                if [[ -z "$matched_shift" ]]; then

                    echo "[keep]"
                    echo "  $version_dir"

                    continue

                fi


                echo "[remove]"
                echo "  config : $cfg"
                echo "  dataset: $dataset"
                echo "  shift  : $matched_shift"
                echo "  path   : $version_dir"

                rm -rf -- "$version_dir"

                local status=$?

                if [[ $status -ne 0 ]]; then

                    echo
                    echo "ERROR: failed to remove:"
                    echo "  $version_dir"

                    return $status

                fi

                ((n_removed += 1))

            done < <(
                find "$dataset_root" \
                    -type d \
                    -name "$version" \
                    -print0
            )


            # -------------------------------------------------------------
            # Remove empty directories left behind by cleanup.
            #
            # Keep the dataset root itself.
            # -------------------------------------------------------------

            find "$dataset_root" \
                -mindepth 1 \
                -depth \
                -type d \
                -empty \
                -delete 2>/dev/null || true

        done
    done


    echo
    echo "======================================================================"
    echo "cf.SelectEvents cleanup completed"
    echo "======================================================================"
    echo "Removed version directories: $n_removed"
    echo

    return 0
}


cleanup_select_events

status=$?

if [[ $status -ne 0 ]]; then

    echo
    echo "ERROR: cf.SelectEvents cleanup failed"
    echo

    exit $status

fi


# ============================================================================
# Final summary
# ============================================================================

echo
echo "======================================================================"
echo "ReduceEvents pipeline completed"
echo "======================================================================"
echo
echo "Successfully produced:"
echo "  - cf.MergeReducedEvents"
echo "  - cf.MergeSelectionStats"
echo "  - cf.MergeSelectionMasks"
echo
echo "Workflow used for every task:"
echo "  $workflow"
echo
echo "Workers used for every task:"
echo "  $workers"
echo
echo "The matching cf.SelectEvents outputs for version:"
echo "  $version"
echo "have been removed."
echo
echo "Cutflow histograms and plots can now use the merged selection masks."
echo
echo "======================================================================"