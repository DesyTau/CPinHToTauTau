#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# Usage
#   ./MSSM_submit_stage_nopoll.sh 22_emu getdatasets
#   ./MSSM_submit_stage_nopoll.sh 22_emu calibrate
#   ./MSSM_submit_stage_nopoll.sh 22_emu select
#   ./MSSM_submit_stage_nopoll.sh 22_emu selection_stats
#   ./MSSM_submit_stage_nopoll.sh 22_emu reduce
#   ./MSSM_submit_stage_nopoll.sh 22_emu reduction_stats
#   ./MSSM_submit_stage_nopoll.sh 22_emu merge_reduced
#   ./MSSM_submit_stage_nopoll.sh 22_emu produce
#   ./MSSM_submit_stage_nopoll.sh 22_emu create_hists
#   ./MSSM_submit_stage_nopoll.sh 22_emu merge_hists
#   ./MSSM_submit_stage_nopoll.sh 22_emu merge_shifted
#
# IMPORTANT:
#   Wait for the previous stage to FINISH before submitting the next one.
#
#   After selection is finished, selection_stats and reduce are independent
#   and can be submitted one after the other without waiting between them.
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
    echo
    echo "Stages:"
    echo "  calibrate"
    echo "  select"
    echo "  selection_stats"
    echo "  reduce"
    echo "  reduction_stats"
    echo "  merge_reduced"
    echo "  produce"
    echo "  create_hists"
    echo "  merge_hists"
    echo "  merge_shifted"
    exit 1
fi


configuration_option="$1"
stage="$2"


# =============================================================================
# Load common configuration
# =============================================================================

script_dir="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${script_dir}/common_run3_MSSM.sh"

set_common_vars "$configuration_option"


# =============================================================================
# Determine the JEC era
# =============================================================================

case "$configuration_option" in

    22_emu)
        jec_era="2022"
        ;;

    22EE_emu)
        jec_era="2022EE"
        ;;

    23_emu)
        jec_era="2023"
        ;;

    23BPix_emu)
        jec_era="2023BPix"
        ;;

    *)
        echo "ERROR: unsupported configuration:"
        echo "  $configuration_option"
        echo
        echo "For no-poll staged production, run each era separately."
        exit 2
        ;;
esac


# =============================================================================
# Local controller settings
# =============================================================================
#
# Only a few Luigi workers are necessary. With --no-poll, they submit one
# remote workflow and then become available for another dataset.
#
# parallel_jobs must be sufficiently large so that a single remote workflow
# is not artificially restricted to the 40-60 jobs configured in law.cfg.
# It does NOT create 10000 lxplus processes.
#
# =============================================================================

workers=1
nopoll_parallel_jobs=10000


# Keep the control files of this submission strategy separate from the
# previous background/signal/data controllers.

export CF_JOB_BASE="${CF_DATA}/jobs/${configuration_option}/${version}/nopoll_stages"

mkdir -p "$CF_JOB_BASE"


# =============================================================================
# Dataset selection
# =============================================================================
#
# All backgrounds + signals + relevant data are submitted together.
#
# data_tau_* is intentionally excluded.
#
# =============================================================================

all_datasets="*"
all_skip_datasets="data_tau_*"

mc_datasets="*"
mc_skip_datasets="data_*"


# =============================================================================
# BDT / inclusive histogram variables
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
# JEC sources
# =============================================================================

jec_sources=(
    "jec_Regrouped_Absolute"
    "jec_Regrouped_BBEC1"
    "jec_Regrouped_EC2"
    "jec_Regrouped_HF"
    "jec_Regrouped_RelativeBal"
    "jec_Regrouped_FlavorQCD"

    "jec_Regrouped_Absolute_${jec_era}"
    "jec_Regrouped_BBEC1_${jec_era}"
    "jec_Regrouped_EC2_${jec_era}"
    "jec_Regrouped_HF_${jec_era}"
    "jec_Regrouped_RelativeSample_${jec_era}"
)


# =============================================================================
# Kinematic shift sources
#
# These require separate event processing.
# =============================================================================

kinematic_shift_sources=(
    "unclustered"
    "recoilresp"
    "recoilres"

    "${jec_sources[@]}"

    "jer"
)


# Convert shift sources into actual shifts:
#
#   nominal
#   source1_up
#   source1_down
#   source2_up
#   source2_down
#   ...
#
# These are used for CalibrateEvents through MergeHistograms.
# =============================================================================

kinematic_shifts=(
    "nominal"
)

for source in "${kinematic_shift_sources[@]}"; do
    kinematic_shifts+=(
        "${source}_up"
        "${source}_down"
    )
done


# =============================================================================
# Weight-only shift sources
#
# These are embedded in the nominal histogram by your histogram producer,
# so they do NOT need independent Calibrate/Select/Reduce/Produce jobs.
# They are nevertheless passed to MergeShiftedHistograms.
# =============================================================================

weight_shift_sources=(
    "muon_weight"
    "electron_weight"
    "Trigger_SF_weight"
    "pu_weight"
    "top_pt_weight"
    "zpt_weight"

    "CMS_PS_ISR"
    "CMS_PS_FSR"
    "CMS_Scale_muR"
    "CMS_Scale_muF"

    "btag_weight_hf"
    "btag_weight_lf"
    "btag_weight_hfstats1"
    "btag_weight_hfstats2"
    "btag_weight_lfstats1"
    "btag_weight_lfstats2"
    "btag_weight_cferr1"
    "btag_weight_cferr2"
)


# =============================================================================
# Complete shift-source list for MergeShiftedHistograms
# =============================================================================

shift_sources=(
    "${weight_shift_sources[@]}"
    "${kinematic_shift_sources[@]}"
)


# =============================================================================
# Convert arrays to comma-separated strings
# =============================================================================

join_by_comma()
{
    local IFS=","
    echo "$*"
}

kinematic_shifts_csv="$(
    join_by_comma "${kinematic_shifts[@]}"
)"

shift_sources_csv="$(
    join_by_comma "${shift_sources[@]}"
)"


# =============================================================================
# Common LAW arguments
# =============================================================================

common_remote_args=(
    --configs "$config"
    --version "$version"

    --workflow "htcondor"

    --workers "$workers"

    --no-poll "True"

    --parallel-jobs "$nopoll_parallel_jobs"
)


common_all_dataset_args=(
    "${common_remote_args[@]}"

    --datasets "$all_datasets"
    --skip-datasets "$all_skip_datasets"
)


common_mc_dataset_args=(
    "${common_remote_args[@]}"

    --datasets "$mc_datasets"
    --skip-datasets "$mc_skip_datasets"
)


# =============================================================================
# Stage submission
# =============================================================================

case "$stage" in


# -----------------------------------------------------------------------------
# 0. Get dataset LFNs
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 0. Get dataset LFNs
#
# Run this before calibration. This creates the LFN lists used by all
# subsequent stages.
# -----------------------------------------------------------------------------

getdatasets)

    law run cf.GetDatasetLFNsWrapper \
        --configs "$config" \
        --datasets "$all_datasets" \
        --skip-datasets "$all_skip_datasets" \
        --shifts "nominal" \
        --workers "$workers"
    ;;

# -----------------------------------------------------------------------------
# 1. Calibration
# -----------------------------------------------------------------------------

calibrate)

    law run cf.CalibrateEventsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrator "main"
    ;;


# -----------------------------------------------------------------------------
# 2. Selection
# -----------------------------------------------------------------------------

select)

    law run cf.SelectEventsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main"
    ;;


# -----------------------------------------------------------------------------
# 3. Merge selection statistics
# -----------------------------------------------------------------------------

selection_stats)

    law run cf.MergeSelectionStatsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main"
    ;;


# -----------------------------------------------------------------------------
# 4. Reduction
# -----------------------------------------------------------------------------

reduce)

    law run cf.ReduceEventsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default"
    ;;


# -----------------------------------------------------------------------------
# 5. Determine reduced-file merging
# -----------------------------------------------------------------------------

reduction_stats)

    law run cf.MergeReductionStatsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default"
    ;;


# -----------------------------------------------------------------------------
# 6. Merge reduced events
# -----------------------------------------------------------------------------

merge_reduced)

    law run cf.MergeReducedEventsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default"
    ;;


# -----------------------------------------------------------------------------
# 7. Produce columns
# -----------------------------------------------------------------------------

produce)

    law run cf.ProduceColumnsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main"
    ;;


# -----------------------------------------------------------------------------
# 8. Create histograms
#
# Keep BDT and inclusive requests separate so their task identities match
# the requirements used later by plotting.
# -----------------------------------------------------------------------------

create_hists)

    echo
    echo "Submitting BDT histograms..."
    echo

    law run cf.CreateHistogramsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$bdt_variable"

    echo
    echo "Submitting inclusive histograms..."
    echo

    law run cf.CreateHistogramsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$inclusive_variable"
    ;;


# -----------------------------------------------------------------------------
# 9. Merge histograms
#
# MC:
#   nominal + kinematic systematics
#
# Data:
#   shift resolution collapses to nominal.
# -----------------------------------------------------------------------------

merge_hists)

    echo
    echo "Submitting BDT histogram merging..."
    echo

    law run cf.MergeHistogramsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$bdt_variable"

    echo
    echo "Submitting inclusive histogram merging..."
    echo

    law run cf.MergeHistogramsWrapper \
        "${common_all_dataset_args[@]}" \
        --shifts "$kinematic_shifts_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$inclusive_variable"
    ;;


# -----------------------------------------------------------------------------
# 10. Merge systematic shifts
#
# MC only.
#
# Data stops at nominal MergeHistograms.
# -----------------------------------------------------------------------------

merge_shifted)

    echo
    echo "Submitting shifted BDT histogram merging..."
    echo

    law run cf.MergeShiftedHistogramsWrapper \
        "${common_mc_dataset_args[@]}" \
        --shift-sources "$shift_sources_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$bdt_variable"

    echo
    echo "Submitting shifted inclusive histogram merging..."
    echo

    law run cf.MergeShiftedHistogramsWrapper \
        "${common_mc_dataset_args[@]}" \
        --shift-sources "$shift_sources_csv" \
        --calibrators "main" \
        --selector "main" \
        --reducer "cf_default" \
        --producers "main" \
        --variables "$inclusive_variable"
    ;;


*)

    echo "ERROR: unknown stage '$stage'"
    exit 3
    ;;

esac