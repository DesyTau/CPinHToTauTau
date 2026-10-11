#!/usr/bin/env bash
set -u
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_SCRIPT="${SCRIPT_DIR}/MSSM_submit_stage.sh"
# --- Defaults ---
PHASE="all"
# Four submit hosts by default. On each host only --shard changes (0,1,2,3).
SHARD=0
N_SHARDS=4
BLOCK_SIZE=4
# Signal blocks must contain one mass only.  The current histogram-variable
# expansion checks that a signal dataset mass matches every requested BDT
# variable mass.  MSSM_submit_stage.sh then filters signal datasets from the
# requested --variables automatically.
SIGNAL_BLOCK_SIZE=1
PARALLEL_JOBS=5
WORKERS=1
WORKFLOW="htcondor"
CAMPAIGN_NAME="bdt_all_masses_v1"
# Shared EOS state: restart markers, logs and cross-host barrier files.
# This is NOT the BDT binning directory.
STATE_BASE="${MSSM_BDT_CAMPAIGN_STATE_DIR:-/eos/project/d/desytau/public/jmalvaso/MSSM_H_tt_store/.mssm_bdt_campaign}"
STATE_DIR=""
# Irregular 2D BDT binning JSONs used by MSSM_H_tt/production/bdt_2d_bins.py.
BDT_BINNING_BASE="${MSSM_BDT_BINNING_BASE:-${MSSM_BDT_2D_BINNING_BASE:-/eos/project/d/desytau/public/jmalvaso/bdt_3_classes_10_features_clippedJetCounts}}"
DRY_RUN=0
FORCE=0
FAIL_FAST=0
USE_PLOT_BARRIER=1
BARRIER_POLL_SECONDS=60
BARRIER_TIMEOUT_SECONDS=86400
PLOT_KIND="nominal"
PLOT_SCOPE="combined"
SIGNAL_MODE="combined"
ERAS_CSV="22_emu,22EE_emu,23_emu,23BPix_emu"
SAMPLE_GROUPS_CSV="data,DY,tt,singlet,other_bkgs,signal"
ONLY_MASSES_CSV=""
EXTRA_STAGE_ARGS=()
N_SELECTED=0
N_SKIPPED_DONE=0
N_SUCCEEDED=0
N_FAILED=0
N_DRY=0
# --- Help ---
usage() {
cat <<'USAGE'
Usage:
  ./MSSM_submit_bdt_campaign.sh <phase> [options] [-- extra-stage-options]
Phases:
  create-hists     Run CreateHistograms in mass blocks
  merge-hists      Run MergeHistograms in the same blocks
  merge-shifted    Run MergeShiftedHistograms in the same blocks
  production       Run create-hists -> merge-hists -> merge-shifted
  plot             Produce final mass-by-mass plots
  all              Run production, synchronize shards, then plot
  status           Show restart-state summary
Core options:
  --shard N                  Zero-based shard index. Default: 0
  --n-shards N               Number of submit-host shards. Default: 4
  --block-size N             Masses per data/background block. Default: 4
  --signal-block-size N      Masses per signal block. Default: 1
  --parallel-jobs N          Remote jobs allowed per LAW graph. Default: 5
  --workers N                Local LAW workers. Default: 1
  --workflow NAME            Workflow backend. Default: htcondor
Campaign selection:
  --eras CSV                 Default: 22_emu,22EE_emu,23_emu,23BPix_emu
  --groups CSV               Default: data,DY,tt,singlet,other_bkgs,signal
  --only-masses CSV          Restrict campaign to explicit masses
  --signal-mode MODE         combined or split. Default: combined
EOS / state:
  --campaign NAME            State namespace. Default: bdt_all_masses_v1
  --state-dir PATH           Explicit shared state/log directory
  --binning-base PATH        EOS base containing BDT binning JSONs
Plot options:
  --plot-kind MODE           nominal, shifted, or both. Default: nominal
  --plot-scope MODE          combined, per-era, or both. Default: combined
Restart/logging:
  --force                    Ignore .done markers and rerun selected work
  --fail-fast                Stop after first failed work unit
Barrier:
  --no-plot-barrier          Do not wait for all production shards before plot
  --barrier-poll N           Poll interval in seconds. Default: 60
  --barrier-timeout N        Timeout in seconds. Default: 86400
Other:
  --dry-run                  Print commands without running them
  -h, --help                 Show this help
Recommended four-host usage:
  host 0:
    ./MSSM_submit_bdt_campaign.sh all --shard 0
  host 1:
    ./MSSM_submit_bdt_campaign.sh all --shard 1
  host 2:
    ./MSSM_submit_bdt_campaign.sh all --shard 2
  host 3:
    ./MSSM_submit_bdt_campaign.sh all --shard 3
All hosts must use the same --campaign, --state-dir, --n-shards, masses,
eras, groups and block sizes. The shared EOS state directory is used for
restart markers, logs and the production->plot barrier.
USAGE
}
# --- Generic helpers ---
die() { echo "ERROR: $*" >&2; exit 1; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
timestamp() { date '+%Y-%m-%d %H:%M:%S %z'; }
sanitize_id() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g'; }
join_by() {
    local delimiter="$1"
    shift
    local out="" item
    for item in "$@"; do
        [[ -n "$out" ]] && out+="$delimiter"
        out+="$item"
    done
    printf '%s' "$out"
}
append_csv() {
    local current="$1"
    local value="$2"
    if [[ -z "$current" ]]; then
        printf '%s' "$value"
    else
        printf '%s,%s' "$current" "$value"
    fi
}
belongs_to_this_shard() {
    local checksum
    checksum="$(printf '%s' "$1" | cksum | awk '{print $1}')"
    (( checksum % N_SHARDS == SHARD ))
}
block_tag() { printf 'M%s' "${1//,/_}"; }
is_signal_group() {
    case "$1" in
        signal|ggphi|bbphi) return 0 ;;
        *) return 1 ;;
    esac
}
# --- BDT mass-block helpers ---
variables_for_block() {
    local block_csv="$1"
    local variables="" mass
    local -a block_masses=()
    IFS=',' read -r -a block_masses <<< "$block_csv"
    for mass in "${block_masses[@]}"; do
        variables="$(append_csv "$variables" "bdt_D_sig_vs_Disc_ggphi_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_sig_vs_Disc_bbphi_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_DY_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_TT_M${mass}")"
    done
    printf '%s' "$variables"
}
make_blocks() {
    local size="$1"
    shift
    local -a input_masses=( "$@" )
    local start end i block
    for ((start=0; start<${#input_masses[@]}; start+=size)); do
        end=$((start + size))
        (( end > ${#input_masses[@]} )) && end=${#input_masses[@]}
        block=""
        for ((i=start; i<end; i++)); do
            block="$(append_csv "$block" "${input_masses[$i]}")"
        done
        printf '%s\n' "$block"
    done
}
# --- State / logs / restartability ---
done_path_for() { printf '%s/done/%s/%s.done' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }
failed_path_for() { printf '%s/failed/%s/%s.failed' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }
log_path_for() { printf '%s/logs/%s/%s.log' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }
ensure_phase_dirs() { mkdir -p "$STATE_DIR/done/$1" "$STATE_DIR/failed/$1" "$STATE_DIR/logs/$1" "$STATE_DIR/barriers"; }
print_command() {
    local arg
    printf '  '
    for arg in "$@"; do
        printf '%s ' "$arg"
    done
    printf '\n'
}
record_success() {
    local phase="$1"
    local item_id="$2"
    local done_path failed_path
    done_path="$(done_path_for "$phase" "$item_id")"
    failed_path="$(failed_path_for "$phase" "$item_id")"
    mkdir -p "$(dirname "$done_path")"
    rm -f "$failed_path"
    {
        echo "status=success"
        echo "time=$(timestamp)"
        echo "host=$(hostname -f 2>/dev/null || hostname)"
        echo "shard=${SHARD}/${N_SHARDS}"
    } > "$done_path"
}
record_failure() {
    local phase="$1"
    local item_id="$2"
    local rc="$3"
    local failed_path done_path
    failed_path="$(failed_path_for "$phase" "$item_id")"
    done_path="$(done_path_for "$phase" "$item_id")"
    mkdir -p "$(dirname "$failed_path")"
    rm -f "$done_path"
    {
        echo "status=failed"
        echo "exit_code=$rc"
        echo "time=$(timestamp)"
        echo "host=$(hostname -f 2>/dev/null || hostname)"
        echo "shard=${SHARD}/${N_SHARDS}"
    } > "$failed_path"
}
run_item() {
    local phase="$1"
    local shard_key="$2"
    local item_id="$3"
    shift 3
    local -a cmd=( "$@" )
    local done_path log_path rc
    belongs_to_this_shard "$shard_key" || return 0
    ((N_SELECTED += 1))
    done_path="$(done_path_for "$phase" "$item_id")"
    log_path="$(log_path_for "$phase" "$item_id")"
    if [[ -f "$done_path" && "$FORCE" -eq 0 ]]; then
        ((N_SKIPPED_DONE += 1))
        echo "[SKIP] [$phase] $item_id"
        return 0
    fi
    if (( DRY_RUN )); then
        ((N_DRY += 1))
        echo "[DRY ] [$phase] $item_id"
        print_command "${cmd[@]}"
        return 0
    fi
    ensure_phase_dirs "$phase"
    echo
    echo "================================================================================"
    echo "[$(timestamp)] [$phase] $item_id"
    echo "Host : $(hostname -f 2>/dev/null || hostname)"
    echo "Shard: ${SHARD}/$((N_SHARDS - 1))"
    echo "Log  : $log_path"
    echo "Command:"
    print_command "${cmd[@]}"
    echo "================================================================================"
    {
        echo "Start : $(timestamp)"
        echo "Phase : $phase"
        echo "Item  : $item_id"
        echo "Host  : $(hostname -f 2>/dev/null || hostname)"
        echo "Shard: ${SHARD}/$((N_SHARDS - 1))"
        echo "Command:"
        print_command "${cmd[@]}"
        echo "================================================================================"
        "${cmd[@]}"
    } 2>&1 | tee "$log_path"
    rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
        record_success "$phase" "$item_id"
        ((N_SUCCEEDED += 1))
        echo "[ OK ] [$phase] $item_id"
    else
        record_failure "$phase" "$item_id" "$rc"
        ((N_FAILED += 1))
        echo "[FAIL] [$phase] $item_id (exit code $rc)" >&2
        echo "       log: $log_path" >&2
        (( FAIL_FAST )) && exit "$rc"
    fi
}
# --- Parse options ---
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    PHASE="$1"
    shift
fi
while (( $# > 0 )); do
    case "$1" in
        --shard) SHARD="$2"; shift 2 ;;
        --shard=*) SHARD="${1#*=}"; shift ;;
        --n-shards) N_SHARDS="$2"; shift 2 ;;
        --n-shards=*) N_SHARDS="${1#*=}"; shift ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --block-size=*) BLOCK_SIZE="${1#*=}"; shift ;;
        --signal-block-size) SIGNAL_BLOCK_SIZE="$2"; shift 2 ;;
        --signal-block-size=*) SIGNAL_BLOCK_SIZE="${1#*=}"; shift ;;
        --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
        --parallel-jobs=*) PARALLEL_JOBS="${1#*=}"; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --workflow) WORKFLOW="$2"; shift 2 ;;
        --workflow=*) WORKFLOW="${1#*=}"; shift ;;
        --campaign) CAMPAIGN_NAME="$2"; shift 2 ;;
        --campaign=*) CAMPAIGN_NAME="${1#*=}"; shift ;;
        --state-dir) STATE_DIR="$2"; shift 2 ;;
        --state-dir=*) STATE_DIR="${1#*=}"; shift ;;
        --binning-base) BDT_BINNING_BASE="$2"; shift 2 ;;
        --binning-base=*) BDT_BINNING_BASE="${1#*=}"; shift ;;
        --eras) ERAS_CSV="$2"; shift 2 ;;
        --eras=*) ERAS_CSV="${1#*=}"; shift ;;
        --groups) SAMPLE_GROUPS_CSV="$2"; shift 2 ;;
        --groups=*) SAMPLE_GROUPS_CSV="${1#*=}"; shift ;;
        --only-masses) ONLY_MASSES_CSV="$2"; shift 2 ;;
        --only-masses=*) ONLY_MASSES_CSV="${1#*=}"; shift ;;
        --signal-mode) SIGNAL_MODE="$2"; shift 2 ;;
        --signal-mode=*) SIGNAL_MODE="${1#*=}"; shift ;;
        --plot-kind) PLOT_KIND="$2"; shift 2 ;;
        --plot-kind=*) PLOT_KIND="${1#*=}"; shift ;;
        --plot-scope) PLOT_SCOPE="$2"; shift 2 ;;
        --plot-scope=*) PLOT_SCOPE="${1#*=}"; shift ;;
        --barrier-poll) BARRIER_POLL_SECONDS="$2"; shift 2 ;;
        --barrier-poll=*) BARRIER_POLL_SECONDS="${1#*=}"; shift ;;
        --barrier-timeout) BARRIER_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --barrier-timeout=*) BARRIER_TIMEOUT_SECONDS="${1#*=}"; shift ;;
        --no-plot-barrier) USE_PLOT_BARRIER=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force) FORCE=1; shift ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --)
            shift
            EXTRA_STAGE_ARGS=( "$@" )
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option '$1'. Use --help."
            ;;
    esac
done
# --- Validate options ---
case "$PHASE" in
    create-hists|merge-hists|merge-shifted|production|plot|all|status) ;;
    *) die "unknown phase '$PHASE'" ;;
esac
is_positive_int "$N_SHARDS" || die "--n-shards must be positive"
is_nonnegative_int "$SHARD" || die "--shard must be non-negative"
(( SHARD < N_SHARDS )) || die "--shard must be smaller than --n-shards"
is_positive_int "$BLOCK_SIZE" || die "--block-size must be positive"
is_positive_int "$SIGNAL_BLOCK_SIZE" || die "--signal-block-size must be positive"
is_positive_int "$PARALLEL_JOBS" || die "--parallel-jobs must be positive"
is_positive_int "$WORKERS" || die "--workers must be positive"
is_positive_int "$BARRIER_POLL_SECONDS" || die "--barrier-poll must be positive"
is_positive_int "$BARRIER_TIMEOUT_SECONDS" || die "--barrier-timeout must be positive"
case "$SIGNAL_MODE" in
    combined|split) ;;
    *) die "--signal-mode must be combined or split" ;;
esac
case "$PLOT_KIND" in
    nominal|shifted|both) ;;
    *) die "--plot-kind must be nominal, shifted or both" ;;
esac
case "$PLOT_SCOPE" in
    combined|per-era|both) ;;
    *) die "--plot-scope must be combined, per-era or both" ;;
esac
[[ -f "$STAGE_SCRIPT" ]] || die "cannot find $STAGE_SCRIPT"
if [[ -z "$STATE_DIR" ]]; then
    STATE_DIR="${STATE_BASE}/${CAMPAIGN_NAME}"
fi
STATE_DIR="$(readlink -m "$STATE_DIR")"
BDT_BINNING_BASE="$(readlink -m "$BDT_BINNING_BASE")"
# Make the EOS binning location explicit for all child processes.
# MSSM_BDT_BINNING_BASE is the canonical variable in the current branch;
# keep MSSM_BDT_2D_BINNING_BASE for backward compatibility.
export MSSM_BDT_BINNING_BASE="$BDT_BINNING_BASE"
export MSSM_BDT_2D_BINNING_BASE="$BDT_BINNING_BASE"
# The campaign itself does not parse the JSON files, but fail early when the
# configured EOS base is unavailable.
[[ -d "$BDT_BINNING_BASE" ]] || die "BDT binning base does not exist: $BDT_BINNING_BASE"
# --- Load configured masses ---
mapfile -t ALL_MASSES < <(
    python - <<'PY'
from MSSM_H_tt.config.mass_points import read_bdt_masses
for mass in read_bdt_masses():
    print(int(mass))
PY
)
(( ${#ALL_MASSES[@]} > 0 )) || die "no BDT masses were loaded"
MASSES=()
if [[ -n "$ONLY_MASSES_CSV" ]]; then
    IFS=',' read -r -a REQUESTED_MASSES <<< "$ONLY_MASSES_CSV"
    for requested in "${REQUESTED_MASSES[@]}"; do
        found=0
        for configured in "${ALL_MASSES[@]}"; do
            if [[ "$requested" == "$configured" ]]; then
                MASSES+=( "$configured" )
                found=1
                break
            fi
        done
        (( found )) || die "requested mass '$requested' is not configured"
    done
else
    MASSES=( "${ALL_MASSES[@]}" )
fi
storage_block_size="$(python - <<'PY'
from MSSM_H_tt.config.mass_points import BDT_HISTOGRAM_STORAGE_BLOCK_SIZE
print(BDT_HISTOGRAM_STORAGE_BLOCK_SIZE)
PY
)" || die "could not load histogram storage block size"
[[ "$BLOCK_SIZE" == "$storage_block_size" ]] || die "--block-size must be ${storage_block_size} to match fixed histogram storage blocks"

# Select whole blocks from the full mass grid, even with --only-masses.
# This makes campaign output reusable by single-variable plots/datacards.
mapfile -t BG_BLOCKS < <(python - "${MASSES[@]}" <<'PY'
import sys
from MSSM_H_tt.config.mass_points import get_bdt_mass_blocks, BDT_HISTOGRAM_STORAGE_BLOCK_SIZE
requested = set(map(int, sys.argv[1:]))
for block in get_bdt_mass_blocks(BDT_HISTOGRAM_STORAGE_BLOCK_SIZE):
    if requested.intersection(block):
        print(",".join(map(str, block)))
PY
)
(( ${#BG_BLOCKS[@]} > 0 )) || die "no background histogram storage blocks were resolved"
mapfile -t SIGNAL_BLOCKS < <(make_blocks "$SIGNAL_BLOCK_SIZE" "${MASSES[@]}")
# --- Eras and sample groups ---
IFS=',' read -r -a ERAS <<< "$ERAS_CSV"
IFS=',' read -r -a REQUESTED_SAMPLE_GROUPS <<< "$SAMPLE_GROUPS_CSV"
# Do NOT call this array GROUPS. GROUPS is a special Bash array containing the
# Unix supplementary group IDs of the current process.
SAMPLE_GROUPS=()
for group in "${REQUESTED_SAMPLE_GROUPS[@]}"; do
    case "$group" in
        data)
            SAMPLE_GROUPS+=( data )
            ;;
        DY|dy)
            SAMPLE_GROUPS+=( DY )
            ;;
        tt)
            SAMPLE_GROUPS+=( tt )
            ;;
        singlet)
            SAMPLE_GROUPS+=( singlet )
            ;;
        other_bkgs)
            SAMPLE_GROUPS+=( other_bkgs )
            ;;
        backgrounds)
            SAMPLE_GROUPS+=( backgrounds )
            ;;
        signal)
            if [[ "$SIGNAL_MODE" == "combined" ]]; then
                SAMPLE_GROUPS+=( signal )
            else
                SAMPLE_GROUPS+=( ggphi bbphi )
            fi
            ;;
        ggphi|bbphi)
            SAMPLE_GROUPS+=( "$group" )
            ;;
        *)
            die "unsupported group '$group'"
            ;;
    esac
done
for era in "${ERAS[@]}"; do
    case "$era" in
        22_emu|22EE_emu|23_emu|23BPix_emu) ;;
        *) die "unsupported era '$era'" ;;
    esac
done
# --- Campaign manifest ---
manifest_content() {
    cat <<EOF_MANIFEST
campaign=${CAMPAIGN_NAME}
n_shards=${N_SHARDS}
masses=$(join_by , "${MASSES[@]}")
eras=$(join_by , "${ERAS[@]}")
groups=$(join_by , "${SAMPLE_GROUPS[@]}")
block_size=${BLOCK_SIZE}
signal_block_size=${SIGNAL_BLOCK_SIZE}
signal_mode=${SIGNAL_MODE}
plot_kind=${PLOT_KIND}
plot_scope=${PLOT_SCOPE}
workflow=${WORKFLOW}
binning_base=${BDT_BINNING_BASE}
EOF_MANIFEST
}
validate_or_create_manifest() {
    (( DRY_RUN )) && return 0
    mkdir -p "$STATE_DIR" "$STATE_DIR/barriers"
    local manifest="$STATE_DIR/campaign.manifest"
    local tmp="$STATE_DIR/.campaign.manifest.$$"
    manifest_content > "$tmp"
    if [[ -f "$manifest" ]]; then
        if ! cmp -s "$manifest" "$tmp"; then
            echo "ERROR: existing campaign state was created with different settings:" >&2
            echo "  $manifest" >&2
            echo >&2
            echo "Use a new --campaign name or an explicit new --state-dir." >&2
            rm -f "$tmp"
            exit 1
        fi
        rm -f "$tmp"
    else
        mv "$tmp" "$manifest"
    fi
}
validate_or_create_manifest
# --- Workflow options ---
append_workflow_options() {
    local phase="$1"
    local -n command_ref="$2"
    case "$phase" in
        create-hists)
            command_ref+=( --create-hists-workflow "$WORKFLOW" )
            ;;
        merge-hists)
command_ref+=( --create-hists-workflow "$WORKFLOW" --merge-hists-workflow "$WORKFLOW" )
            ;;
        merge-shifted|plot1d|plot1d-shifted)
command_ref+=( --create-hists-workflow "$WORKFLOW" --merge-hists-workflow "$WORKFLOW" --merge-shifted-workflow "$WORKFLOW" )
            ;;
    esac
}
# --- Production ---
run_production_phase() {
    local phase="$1"
    local era group block vars tag shard_key item_id
    local -a blocks=()
    local -a cmd=()
    echo
    echo "### Production phase: $phase"
    for era in "${ERAS[@]}"; do
        for group in "${SAMPLE_GROUPS[@]}"; do
            if is_signal_group "$group"; then
                # One mass per signal work item. MSSM_submit_stage.sh uses the
                # requested --variables below to select only ggphi/bbphi samples
                # with the matching physical signal mass.
                blocks=( "${SIGNAL_BLOCKS[@]}" )
            else
                blocks=( "${BG_BLOCKS[@]}" )
            fi
            for block in "${blocks[@]}"; do
                vars="$(variables_for_block "$block")"
                tag="$(block_tag "$block")"
                # Use one stable key for all production phases so that the same
                # era/group/mass block stays on the same submit host.
                shard_key="production|${era}|${group}|${tag}"
                item_id="${era}__${group}__${tag}"
cmd=( "$STAGE_SCRIPT" "$phase" "$era" "$group" --variables "$vars" --producers main --parallel-jobs "$PARALLEL_JOBS" --workers "$WORKERS" )
                append_workflow_options "$phase" cmd
                if (( ${#EXTRA_STAGE_ARGS[@]} > 0 )); then
                    cmd+=( "${EXTRA_STAGE_ARGS[@]}" )
                fi
                run_item "$phase" "$shard_key" "$item_id" "${cmd[@]}"
            done
        done
    done
}
# --- Plotting ---
plot_targets() {
    case "$PLOT_SCOPE" in
        combined)
            printf '%s\n' 22and23_emu
            ;;
        per-era)
            printf '%s\n' "${ERAS[@]}"
            ;;
        both)
            printf '%s\n' 22and23_emu
            printf '%s\n' "${ERAS[@]}"
            ;;
    esac
}
plot_stages() {
    case "$PLOT_KIND" in
        nominal)
            printf '%s\n' plot1d
            ;;
        shifted)
            printf '%s\n' plot1d-shifted
            ;;
        both)
            printf '%s\n' plot1d plot1d-shifted
            ;;
    esac
}
run_plot_phase() {
    local plot_stage target mass disc variable shard_key item_id
    local -a cmd=()
    echo
    echo "### Final plotting"
    while IFS= read -r plot_stage; do
        while IFS= read -r target; do
            for mass in "${MASSES[@]}"; do
                for disc in D_sig_vs_Disc_ggphi D_sig_vs_Disc_bbphi D_DY D_TT; do
                    variable="bdt_${disc}_M${mass}"
                    shard_key="plot|${plot_stage}|${target}|${variable}"
                    item_id="${plot_stage}__${target}__${variable}"
cmd=( "$STAGE_SCRIPT" "$plot_stage" "$target" datacard "$variable" --producers main --parallel-jobs "$PARALLEL_JOBS" --workers "$WORKERS" )
                    append_workflow_options "$plot_stage" cmd
                    if (( ${#EXTRA_STAGE_ARGS[@]} > 0 )); then
                        cmd+=( "${EXTRA_STAGE_ARGS[@]}" )
                    fi
                    run_item plot "$shard_key" "$item_id" "${cmd[@]}"
                done
            done
        done < <(plot_targets)
    done < <(plot_stages)
}
# --- Cross-host barrier ---
production_barrier_path() { printf '%s/barriers/production_shard_%s.done' "$STATE_DIR" "$1"; }
mark_production_shard_complete() {
    local marker
    marker="$(production_barrier_path "$SHARD")"
    mkdir -p "$STATE_DIR/barriers"
    {
        echo "status=success"
        echo "time=$(timestamp)"
        echo "host=$(hostname -f 2>/dev/null || hostname)"
        echo "shard=${SHARD}/${N_SHARDS}"
    } > "$marker"
    echo "[BARRIER] production shard ${SHARD}/${N_SHARDS} complete"
}
all_production_shards_complete() {
    local i
    for ((i=0; i<N_SHARDS; i++)); do
        [[ -f "$(production_barrier_path "$i")" ]] || return 1
    done
    return 0
}
wait_for_production_barrier() {
    (( N_SHARDS > 1 )) || return 0
    (( USE_PLOT_BARRIER )) || return 0
    (( DRY_RUN )) && return 0
    local start now elapsed i missing
    start="$(date +%s)"
    while ! all_production_shards_complete; do
        now="$(date +%s)"
        elapsed=$((now - start))
        if (( elapsed >= BARRIER_TIMEOUT_SECONDS )); then
            echo "ERROR: timed out waiting for production shards" >&2
            return 1
        fi
        missing=""
        for ((i=0; i<N_SHARDS; i++)); do
            if [[ ! -f "$(production_barrier_path "$i")" ]]; then
                missing+=" $i"
            fi
        done
        echo "[BARRIER] waiting for production shards:${missing}"
        sleep "$BARRIER_POLL_SECONDS"
    done
    echo "[BARRIER] all ${N_SHARDS} production shards are complete"
}
# --- Status ---
status_one() {
    if [[ -f "$(done_path_for "$1" "$2")" ]]; then
        printf done
        return
    fi
    if [[ -f "$(failed_path_for "$1" "$2")" ]]; then
        printf failed
        return
    fi
    printf pending
}
show_status() {
    local phase era group block tag item_id state
    local n_done n_failed n_pending total
    local -a blocks=()
    echo "Campaign : $CAMPAIGN_NAME"
    echo "State dir: $STATE_DIR"
    echo "Binning  : $BDT_BINNING_BASE"
    echo "Masses   : $(join_by , "${MASSES[@]}")"
    echo "Eras     : $(join_by , "${ERAS[@]}")"
    echo "Groups   : $(join_by , "${SAMPLE_GROUPS[@]}")"
    echo
    for phase in create-hists merge-hists merge-shifted; do
        n_done=0
        n_failed=0
        n_pending=0
        total=0
        for era in "${ERAS[@]}"; do
            for group in "${SAMPLE_GROUPS[@]}"; do
                if is_signal_group "$group"; then
                    blocks=( "${SIGNAL_BLOCKS[@]}" )
                else
                    blocks=( "${BG_BLOCKS[@]}" )
                fi
                for block in "${blocks[@]}"; do
                    tag="$(block_tag "$block")"
                    item_id="${era}__${group}__${tag}"
                    state="$(status_one "$phase" "$item_id")"
                    ((total += 1))
                    case "$state" in
                        done) ((n_done += 1)) ;;
                        failed) ((n_failed += 1)) ;;
                        pending) ((n_pending += 1)) ;;
                    esac
                done
            done
        done
        printf '%-15s total=%4d done=%4d failed=%4d pending=%4d\n' "$phase" "$total" "$n_done" "$n_failed" "$n_pending"
    done
    n_done=0
    n_failed=0
    n_pending=0
    total=0
    local plot_stage target mass disc variable
    while IFS= read -r plot_stage; do
        while IFS= read -r target; do
            for mass in "${MASSES[@]}"; do
                for disc in D_sig_vs_Disc_ggphi D_sig_vs_Disc_bbphi D_DY D_TT; do
                    variable="bdt_${disc}_M${mass}"
                    item_id="${plot_stage}__${target}__${variable}"
                    state="$(status_one plot "$item_id")"
                    ((total += 1))
                    case "$state" in
                        done) ((n_done += 1)) ;;
                        failed) ((n_failed += 1)) ;;
                        pending) ((n_pending += 1)) ;;
                    esac
                done
            done
        done < <(plot_targets)
    done < <(plot_stages)
    printf '%-15s total=%4d done=%4d failed=%4d pending=%4d\n' plot "$total" "$n_done" "$n_failed" "$n_pending"
    echo
    echo "Production barriers:"
    local i
    for ((i=0; i<N_SHARDS; i++)); do
        if [[ -f "$(production_barrier_path "$i")" ]]; then
            echo "  shard $i: done"
        else
            echo "  shard $i: pending"
        fi
    done
    echo
    echo "Failed work items:"
    if find "$STATE_DIR/failed" -type f -name '*.failed' -print -quit 2>/dev/null | grep -q .; then
        find "$STATE_DIR/failed" -type f -name '*.failed' -print 2>/dev/null | sort
    else
        echo "  none"
    fi
}
# --- Summary ---
print_summary() {
    echo
    echo "================================================================================"
    echo "Campaign      : $CAMPAIGN_NAME"
    echo "Phase         : $PHASE"
    echo "Shard         : ${SHARD}/$((N_SHARDS - 1))"
    echo "Masses        : $(join_by , "${MASSES[@]}")"
    echo "Eras          : $(join_by , "${ERAS[@]}")"
    echo "Groups        : $(join_by , "${SAMPLE_GROUPS[@]}")"
    echo "Block size    : $BLOCK_SIZE"
    echo "Signal block  : $SIGNAL_BLOCK_SIZE"
    echo "Parallel jobs : $PARALLEL_JOBS"
    echo "Workers       : $WORKERS"
    echo "Workflow      : $WORKFLOW"
    echo "BDT binning   : $BDT_BINNING_BASE"
    echo "State dir     : $STATE_DIR"
    echo "Selected      : $N_SELECTED"
    echo "Skipped done  : $N_SKIPPED_DONE"
    echo "Succeeded     : $N_SUCCEEDED"
    echo "Failed        : $N_FAILED"
    echo "Dry-run items : $N_DRY"
    echo "================================================================================"
}
# --- Start banner ---
echo "================================================================================"
echo "MSSM BDT campaign"
echo "Phase         : $PHASE"
echo "Campaign      : $CAMPAIGN_NAME"
echo "Shard         : ${SHARD}/${N_SHARDS}"
echo "Masses        : $(join_by , "${MASSES[@]}")"
echo "Eras          : $(join_by , "${ERAS[@]}")"
echo "Groups        : $(join_by , "${SAMPLE_GROUPS[@]}")"
echo "Block size    : $BLOCK_SIZE"
echo "Signal block  : $SIGNAL_BLOCK_SIZE"
echo "Parallel jobs : $PARALLEL_JOBS"
echo "Workers       : $WORKERS"
echo "Workflow      : $WORKFLOW"
echo "Plot kind     : $PLOT_KIND"
echo "Plot scope    : $PLOT_SCOPE"
echo "BDT binning   : $BDT_BINNING_BASE"
echo "State dir     : $STATE_DIR"
echo "Dry run       : $DRY_RUN"
echo "================================================================================"
# --- Execute ---
case "$PHASE" in
    create-hists)
        run_production_phase create-hists
        ;;
    merge-hists)
        run_production_phase merge-hists
        ;;
    merge-shifted)
        run_production_phase merge-shifted
        ;;
    production)
        run_production_phase create-hists
        run_production_phase merge-hists
        run_production_phase merge-shifted
        if (( N_FAILED == 0 && DRY_RUN == 0 )); then
            mark_production_shard_complete
        fi
        ;;
    plot)
        run_plot_phase
        ;;
    all)
        run_production_phase create-hists
        run_production_phase merge-hists
        run_production_phase merge-shifted
        if (( N_FAILED != 0 )); then
            echo "Production has failures; plotting will not start." >&2
            print_summary
            exit 1
        fi
        if (( DRY_RUN == 0 )); then
            mark_production_shard_complete
            wait_for_production_barrier || {
                print_summary
                exit 1
            }
        fi
        run_plot_phase
        ;;
    status)
        show_status
        exit 0
        ;;
esac
print_summary
(( N_FAILED > 0 )) && exit 1
exit 0