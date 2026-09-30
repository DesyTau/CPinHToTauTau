#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_SCRIPT="${SCRIPT_DIR}/MSSM_submit_stage.sh"

PHASE="all"
SHARD=0
N_SHARDS=1
BLOCK_SIZE=4
SIGNAL_BLOCK_SIZE=4
PARALLEL_JOBS=5
WORKERS=1
WORKFLOW="htcondor"
CAMPAIGN_NAME="bdt_all_masses_v1"
STATE_BASE="${MSSM_BDT_CAMPAIGN_STATE_DIR:-${HOME}/.cache/mssm_bdt_campaign}"
STATE_DIR=""
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
GROUPS_CSV="data,DY,tt,singlet,other_bkgs,signal"
ONLY_MASSES_CSV=""
EXTRA_STAGE_ARGS=()
N_SELECTED=0
N_SKIPPED_DONE=0
N_SUCCEEDED=0
N_FAILED=0
N_DRY=0

usage() {
cat <<'USAGE'
Usage:
  ./MSSM_submit_bdt_campaign.sh <phase> [options] [-- extra-stage-options]

Phases:
  create-hists     Run CreateHistograms in small production blocks
  merge-hists      Run MergeHistograms in the same blocks
  merge-shifted    Run MergeShiftedHistograms in the same blocks
  production       Run create-hists -> merge-hists -> merge-shifted
  plot             Produce final mass-by-mass plots
  all              Run production, synchronize shards, then plot
  status           Show restart-state summary

Core options:
  --shard N                  Zero-based shard index. Default: 0
  --n-shards N               Number of submit-host shards. Default: 1
  --block-size N             Masses per data/background block. Default: 4
  --signal-block-size N      Masses per signal block. Default: 4
  --parallel-jobs N          Remote jobs allowed per LAW graph. Default: 5
  --workers N                Local LAW workers. Default: 1
  --workflow NAME            Workflow backend. Default: htcondor

Campaign selection:
  --eras CSV                 Default: 22_emu,22EE_emu,23_emu,23BPix_emu
  --groups CSV               Default: data,DY,tt,singlet,other_bkgs,signal
  --only-masses CSV          Restrict campaign to explicit masses
  --signal-mode MODE         combined or split. Default: combined

Plot options:
  --plot-kind MODE           nominal, shifted, or both. Default: nominal
  --plot-scope MODE          combined, per-era, or both. Default: combined

Restart/logging:
  --campaign NAME            State namespace. Default: bdt_all_masses_v1
  --state-dir PATH           Explicit shared state/log directory
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
    ./MSSM_submit_bdt_campaign.sh all --shard 0 --n-shards 4
  host 1:
    ./MSSM_submit_bdt_campaign.sh all --shard 1 --n-shards 4
  host 2:
    ./MSSM_submit_bdt_campaign.sh all --shard 2 --n-shards 4
  host 3:
    ./MSSM_submit_bdt_campaign.sh all --shard 3 --n-shards 4

Use the same --campaign, --state-dir and --n-shards on every host.
The automatic plot barrier requires a state directory visible from every host.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
timestamp() { date '+%Y-%m-%d %H:%M:%S %z'; }
sanitize_id() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g'; }

join_by() {
    local delimiter="$1"; shift
    local out="" item
    for item in "$@"; do
        [[ -n "$out" ]] && out+="$delimiter"
        out+="$item"
    done
    printf '%s' "$out"
}

append_csv() {
    local current="$1" value="$2"
    [[ -z "$current" ]] && printf '%s' "$value" || printf '%s,%s' "$current" "$value"
}

belongs_to_this_shard() {
    local checksum
    checksum="$(printf '%s' "$1" | cksum | awk '{print $1}')"
    (( checksum % N_SHARDS == SHARD ))
}

block_tag() { printf 'M%s' "${1//,/_}"; }

variables_for_block() {
    local block_csv="$1" variables="" mass
    local masses=()
    IFS=',' read -r -a masses <<< "$block_csv"
    for mass in "${masses[@]}"; do
        variables="$(append_csv "$variables" "bdt_D_sig_vs_Disc_ggphi_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_sig_vs_Disc_bbphi_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_DY_M${mass}")"
        variables="$(append_csv "$variables" "bdt_D_TT_M${mass}")"
    done
    printf '%s' "$variables"
}

done_path_for() { printf '%s/done/%s/%s.done' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }
failed_path_for() { printf '%s/failed/%s/%s.failed' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }
log_path_for() { printf '%s/logs/%s/%s.log' "$STATE_DIR" "$1" "$(sanitize_id "$2")"; }

ensure_phase_dirs() {
    mkdir -p "$STATE_DIR/done/$1" "$STATE_DIR/failed/$1" "$STATE_DIR/logs/$1" "$STATE_DIR/barriers"
}

print_command() {
    local arg
    printf '  '
    for arg in "$@"; do printf '%q ' "$arg"; done
    printf '\n'
}

record_success() {
    local done_path failed_path
    done_path="$(done_path_for "$1" "$2")"
    failed_path="$(failed_path_for "$1" "$2")"
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
    local failed_path done_path
    failed_path="$(failed_path_for "$1" "$2")"
    done_path="$(done_path_for "$1" "$2")"
    mkdir -p "$(dirname "$failed_path")"
    rm -f "$done_path"
    {
        echo "status=failed"
        echo "exit_code=$3"
        echo "time=$(timestamp)"
        echo "host=$(hostname -f 2>/dev/null || hostname)"
        echo "shard=${SHARD}/${N_SHARDS}"
    } > "$failed_path"
}

run_item() {
    local phase="$1" shard_key="$2" item_id="$3"
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
    echo "Shard: ${SHARD}/${N_SHARDS}"
    echo "Log  : $log_path"
    echo "Command:"
    print_command "${cmd[@]}"
    echo "================================================================================"

    {
        echo "Start : $(timestamp)"
        echo "Phase : $phase"
        echo "Item  : $item_id"
        echo "Host  : $(hostname -f 2>/dev/null || hostname)"
        echo "Shard : ${SHARD}/${N_SHARDS}"
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

if (( $# > 0 )) && [[ "$1" != -* ]]; then PHASE="$1"; shift; fi

while (( $# > 0 )); do
    case "$1" in
        --shard) SHARD="$2"; shift 2;; --shard=*) SHARD="${1#*=}"; shift;;
        --n-shards) N_SHARDS="$2"; shift 2;; --n-shards=*) N_SHARDS="${1#*=}"; shift;;
        --block-size) BLOCK_SIZE="$2"; shift 2;; --block-size=*) BLOCK_SIZE="${1#*=}"; shift;;
        --signal-block-size) SIGNAL_BLOCK_SIZE="$2"; shift 2;; --signal-block-size=*) SIGNAL_BLOCK_SIZE="${1#*=}"; shift;;
        --parallel-jobs) PARALLEL_JOBS="$2"; shift 2;; --parallel-jobs=*) PARALLEL_JOBS="${1#*=}"; shift;;
        --workers) WORKERS="$2"; shift 2;; --workers=*) WORKERS="${1#*=}"; shift;;
        --workflow) WORKFLOW="$2"; shift 2;; --workflow=*) WORKFLOW="${1#*=}"; shift;;
        --campaign) CAMPAIGN_NAME="$2"; shift 2;; --campaign=*) CAMPAIGN_NAME="${1#*=}"; shift;;
        --state-dir) STATE_DIR="$2"; shift 2;; --state-dir=*) STATE_DIR="${1#*=}"; shift;;
        --eras) ERAS_CSV="$2"; shift 2;; --eras=*) ERAS_CSV="${1#*=}"; shift;;
        --groups) GROUPS_CSV="$2"; shift 2;; --groups=*) GROUPS_CSV="${1#*=}"; shift;;
        --only-masses) ONLY_MASSES_CSV="$2"; shift 2;; --only-masses=*) ONLY_MASSES_CSV="${1#*=}"; shift;;
        --signal-mode) SIGNAL_MODE="$2"; shift 2;; --signal-mode=*) SIGNAL_MODE="${1#*=}"; shift;;
        --plot-kind) PLOT_KIND="$2"; shift 2;; --plot-kind=*) PLOT_KIND="${1#*=}"; shift;;
        --plot-scope) PLOT_SCOPE="$2"; shift 2;; --plot-scope=*) PLOT_SCOPE="${1#*=}"; shift;;
        --barrier-poll) BARRIER_POLL_SECONDS="$2"; shift 2;; --barrier-poll=*) BARRIER_POLL_SECONDS="${1#*=}"; shift;;
        --barrier-timeout) BARRIER_TIMEOUT_SECONDS="$2"; shift 2;; --barrier-timeout=*) BARRIER_TIMEOUT_SECONDS="${1#*=}"; shift;;
        --no-plot-barrier) USE_PLOT_BARRIER=0; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --force) FORCE=1; shift;;
        --fail-fast) FAIL_FAST=1; shift;;
        --) shift; EXTRA_STAGE_ARGS=( "$@" ); break;;
        -h|--help) usage; exit 0;;
        *) die "unknown option '$1'. Use --help.";;
    esac
done

case "$PHASE" in create-hists|merge-hists|merge-shifted|production|plot|all|status) ;; *) die "unknown phase '$PHASE'";; esac
is_positive_int "$N_SHARDS" || die "--n-shards must be positive"
is_nonnegative_int "$SHARD" || die "--shard must be non-negative"
(( SHARD < N_SHARDS )) || die "--shard must be smaller than --n-shards"
is_positive_int "$BLOCK_SIZE" || die "--block-size must be positive"
is_positive_int "$SIGNAL_BLOCK_SIZE" || die "--signal-block-size must be positive"
is_positive_int "$PARALLEL_JOBS" || die "--parallel-jobs must be positive"
is_positive_int "$WORKERS" || die "--workers must be positive"
is_positive_int "$BARRIER_POLL_SECONDS" || die "--barrier-poll must be positive"
is_positive_int "$BARRIER_TIMEOUT_SECONDS" || die "--barrier-timeout must be positive"
case "$SIGNAL_MODE" in combined|split) ;; *) die "--signal-mode must be combined or split";; esac
case "$PLOT_KIND" in nominal|shifted|both) ;; *) die "--plot-kind must be nominal, shifted or both";; esac
case "$PLOT_SCOPE" in combined|per-era|both) ;; *) die "--plot-scope must be combined, per-era or both";; esac
[[ -f "$STAGE_SCRIPT" ]] || die "cannot find $STAGE_SCRIPT"
[[ -z "$STATE_DIR" ]] && STATE_DIR="${STATE_BASE}/${CAMPAIGN_NAME}"
STATE_DIR="$(readlink -m "$STATE_DIR")"
mkdir -p "$STATE_DIR"

mapfile -t ALL_MASSES < <(python - <<'PY'
from MSSM_H_tt.config.mass_points import read_bdt_masses
for m in read_bdt_masses():
    print(int(m))
PY
)
(( ${#ALL_MASSES[@]} > 0 )) || die "no BDT masses were loaded"

MASSES=()
if [[ -n "$ONLY_MASSES_CSV" ]]; then
    IFS=',' read -r -a REQUESTED_MASSES <<< "$ONLY_MASSES_CSV"
    for requested in "${REQUESTED_MASSES[@]}"; do
        found=0
        for configured in "${ALL_MASSES[@]}"; do
            if [[ "$requested" == "$configured" ]]; then MASSES+=( "$configured" ); found=1; break; fi
        done
        (( found )) || die "requested mass '$requested' is not configured"
    done
else
    MASSES=( "${ALL_MASSES[@]}" )
fi

make_blocks() {
    local size="$1"; shift
    local -a masses=( "$@" )
    local start end i block
    for ((start=0; start<${#masses[@]}; start+=size)); do
        end=$((start + size)); (( end > ${#masses[@]} )) && end=${#masses[@]}
        block=""
        for ((i=start; i<end; i++)); do block="$(append_csv "$block" "${masses[$i]}")"; done
        printf '%s\n' "$block"
    done
}

mapfile -t BG_BLOCKS < <(make_blocks "$BLOCK_SIZE" "${MASSES[@]}")
mapfile -t SIGNAL_BLOCKS < <(make_blocks "$SIGNAL_BLOCK_SIZE" "${MASSES[@]}")
IFS=',' read -r -a ERAS <<< "$ERAS_CSV"
IFS=',' read -r -a REQUESTED_GROUPS <<< "$GROUPS_CSV"
GROUPS=()
for group in "${REQUESTED_GROUPS[@]}"; do
    case "$group" in
        data|DY|dy|tt|singlet|other_bkgs|backgrounds) GROUPS+=( "$group" );;
        signal) [[ "$SIGNAL_MODE" == combined ]] && GROUPS+=( signal ) || GROUPS+=( ggphi bbphi );;
        ggphi|bbphi) GROUPS+=( "$group" );;
        *) die "unsupported group '$group'";;
    esac
done
for era in "${ERAS[@]}"; do
    case "$era" in 22_emu|22EE_emu|23_emu|23BPix_emu) ;; *) die "unsupported era '$era'";; esac
done

append_workflow_options() {
    local phase="$1"; local -n c="$2"
    case "$phase" in
        create-hists) c+=( --create-hists-workflow "$WORKFLOW" );;
        merge-hists) c+=( --create-hists-workflow "$WORKFLOW" --merge-hists-workflow "$WORKFLOW" );;
        merge-shifted|plot1d|plot1d-shifted)
            c+=( --create-hists-workflow "$WORKFLOW" --merge-hists-workflow "$WORKFLOW" --merge-shifted-workflow "$WORKFLOW" );;
    esac
}

run_production_phase() {
    local phase="$1" era group block vars tag shard_key item_id
    local -a blocks cmd
    echo; echo "### Production phase: $phase"
    for era in "${ERAS[@]}"; do
        for group in "${GROUPS[@]}"; do
            case "$group" in signal|ggphi|bbphi) blocks=( "${SIGNAL_BLOCKS[@]}" );; *) blocks=( "${BG_BLOCKS[@]}" );; esac
            for block in "${blocks[@]}"; do
                vars="$(variables_for_block "$block")"
                tag="$(block_tag "$block")"
                shard_key="production|${era}|${group}|${tag}"
                item_id="${era}__${group}__${tag}"
                cmd=( "$STAGE_SCRIPT" "$phase" "$era" "$group" --variables "$vars" --producers main --parallel-jobs "$PARALLEL_JOBS" --workers "$WORKERS" )
                append_workflow_options "$phase" cmd
                (( ${#EXTRA_STAGE_ARGS[@]} > 0 )) && cmd+=( "${EXTRA_STAGE_ARGS[@]}" )
                run_item "$phase" "$shard_key" "$item_id" "${cmd[@]}"
            done
        done
    done
}

plot_targets() {
    case "$PLOT_SCOPE" in
        combined) printf '%s\n' 22and23_emu;;
        per-era) printf '%s\n' "${ERAS[@]}";;
        both) printf '%s\n' 22and23_emu; printf '%s\n' "${ERAS[@]}";;
    esac
}

plot_stages() {
    case "$PLOT_KIND" in
        nominal) printf '%s\n' plot1d;;
        shifted) printf '%s\n' plot1d-shifted;;
        both) printf '%s\n' plot1d plot1d-shifted;;
    esac
}

run_plot_phase() {
    local plot_stage target mass disc variable shard_key item_id
    local -a cmd
    echo; echo "### Final plotting"
    while IFS= read -r plot_stage; do
        while IFS= read -r target; do
            for mass in "${MASSES[@]}"; do
                for disc in D_sig_vs_Disc_ggphi D_sig_vs_Disc_bbphi D_DY D_TT; do
                    variable="bdt_${disc}_M${mass}"
                    shard_key="plot|${plot_stage}|${target}|${variable}"
                    item_id="${plot_stage}__${target}__${variable}"
                    cmd=( "$STAGE_SCRIPT" "$plot_stage" "$target" datacard "$variable" --producers main --parallel-jobs "$PARALLEL_JOBS" --workers "$WORKERS" )
                    append_workflow_options "$plot_stage" cmd
                    (( ${#EXTRA_STAGE_ARGS[@]} > 0 )) && cmd+=( "${EXTRA_STAGE_ARGS[@]}" )
                    run_item plot "$shard_key" "$item_id" "${cmd[@]}"
                done
            done
        done < <(plot_targets)
    done < <(plot_stages)
}

production_barrier_path() { printf '%s/barriers/production_shard_%s.done' "$STATE_DIR" "$1"; }
mark_production_shard_complete() {
    local marker="$(production_barrier_path "$SHARD")"
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
    for ((i=0; i<N_SHARDS; i++)); do [[ -f "$(production_barrier_path "$i")" ]] || return 1; done
    return 0
}
wait_for_production_barrier() {
    (( N_SHARDS > 1 )) || return 0
    (( USE_PLOT_BARRIER )) || return 0
    (( DRY_RUN )) && return 0
    local start="$(date +%s)" now elapsed i missing
    while ! all_production_shards_complete; do
        now="$(date +%s)"; elapsed=$((now - start))
        if (( elapsed >= BARRIER_TIMEOUT_SECONDS )); then
            echo "ERROR: timed out waiting for production shards" >&2
            return 1
        fi
        missing=""
        for ((i=0; i<N_SHARDS; i++)); do [[ -f "$(production_barrier_path "$i")" ]] || missing+=" $i"; done
        echo "[BARRIER] waiting for production shards:$missing"
        sleep "$BARRIER_POLL_SECONDS"
    done
    echo "[BARRIER] all ${N_SHARDS} production shards are complete"
}

status_one() {
    [[ -f "$(done_path_for "$1" "$2")" ]] && { printf done; return; }
    [[ -f "$(failed_path_for "$1" "$2")" ]] && { printf failed; return; }
    printf pending
}

show_status() {
    local phase era group block tag item_id state done failed pending total
    local -a blocks
    echo "Campaign : $CAMPAIGN_NAME"
    echo "State dir: $STATE_DIR"
    echo "Masses   : $(join_by , "${MASSES[@]}")"
    echo
    for phase in create-hists merge-hists merge-shifted; do
        done=0; failed=0; pending=0; total=0
        for era in "${ERAS[@]}"; do
            for group in "${GROUPS[@]}"; do
                case "$group" in signal|ggphi|bbphi) blocks=( "${SIGNAL_BLOCKS[@]}" );; *) blocks=( "${BG_BLOCKS[@]}" );; esac
                for block in "${blocks[@]}"; do
                    tag="$(block_tag "$block")"; item_id="${era}__${group}__${tag}"; state="$(status_one "$phase" "$item_id")"; ((total+=1))
                    case "$state" in done) ((done+=1));; failed) ((failed+=1));; pending) ((pending+=1));; esac
                done
            done
        done
        printf '%-15s total=%4d done=%4d failed=%4d pending=%4d\n' "$phase" "$total" "$done" "$failed" "$pending"
    done
    done=0; failed=0; pending=0; total=0
    local plot_stage target mass disc variable
    while IFS= read -r plot_stage; do
        while IFS= read -r target; do
            for mass in "${MASSES[@]}"; do
                for disc in D_sig_vs_Disc_ggphi D_sig_vs_Disc_bbphi D_DY D_TT; do
                    variable="bdt_${disc}_M${mass}"; item_id="${plot_stage}__${target}__${variable}"; state="$(status_one plot "$item_id")"; ((total+=1))
                    case "$state" in done) ((done+=1));; failed) ((failed+=1));; pending) ((pending+=1));; esac
                done
            done
        done < <(plot_targets)
    done < <(plot_stages)
    printf '%-15s total=%4d done=%4d failed=%4d pending=%4d\n' plot "$total" "$done" "$failed" "$pending"
    echo; echo "Production barriers:"
    for ((i=0; i<N_SHARDS; i++)); do [[ -f "$(production_barrier_path "$i")" ]] && echo "  shard $i: done" || echo "  shard $i: pending"; done
    echo; echo "Failed work items:"
    if find "$STATE_DIR/failed" -type f -name '*.failed' -print -quit 2>/dev/null | grep -q .; then
        find "$STATE_DIR/failed" -type f -name '*.failed' -print 2>/dev/null | sort
    else
        echo "  none"
    fi
}

print_summary() {
    echo
    echo "================================================================================"
    echo "Campaign      : $CAMPAIGN_NAME"
    echo "Phase         : $PHASE"
    echo "Shard         : ${SHARD}/${N_SHARDS}"
    echo "Masses        : $(join_by , "${MASSES[@]}")"
    echo "Block size    : $BLOCK_SIZE"
    echo "Signal block  : $SIGNAL_BLOCK_SIZE"
    echo "Parallel jobs : $PARALLEL_JOBS"
    echo "Workers       : $WORKERS"
    echo "State dir     : $STATE_DIR"
    echo "Selected      : $N_SELECTED"
    echo "Skipped done  : $N_SKIPPED_DONE"
    echo "Succeeded     : $N_SUCCEEDED"
    echo "Failed        : $N_FAILED"
    echo "Dry-run items : $N_DRY"
    echo "================================================================================"
}

echo "================================================================================"
echo "MSSM BDT campaign"
echo "Phase         : $PHASE"
echo "Campaign      : $CAMPAIGN_NAME"
echo "Shard         : ${SHARD}/${N_SHARDS}"
echo "Masses        : $(join_by , "${MASSES[@]}")"
echo "Eras          : $(join_by , "${ERAS[@]}")"
echo "Groups        : $(join_by , "${GROUPS[@]}")"
echo "Block size    : $BLOCK_SIZE"
echo "Signal block  : $SIGNAL_BLOCK_SIZE"
echo "Parallel jobs : $PARALLEL_JOBS"
echo "Workers       : $WORKERS"
echo "Workflow      : $WORKFLOW"
echo "Plot kind     : $PLOT_KIND"
echo "Plot scope    : $PLOT_SCOPE"
echo "State dir     : $STATE_DIR"
echo "Dry run       : $DRY_RUN"
echo "================================================================================"

case "$PHASE" in
    create-hists) run_production_phase create-hists;;
    merge-hists) run_production_phase merge-hists;;
    merge-shifted) run_production_phase merge-shifted;;
    production)
        run_production_phase create-hists
        run_production_phase merge-hists
        run_production_phase merge-shifted
        (( N_FAILED == 0 && DRY_RUN == 0 )) && mark_production_shard_complete
        ;;
    plot) run_plot_phase;;
    all)
        run_production_phase create-hists
        run_production_phase merge-hists
        run_production_phase merge-shifted
        if (( N_FAILED != 0 )); then echo "Production has failures; plotting will not start." >&2; print_summary; exit 1; fi
        if (( DRY_RUN == 0 )); then
            mark_production_shard_complete
            wait_for_production_barrier || { print_summary; exit 1; }
        fi
        run_plot_phase
        ;;
    status) show_status; exit 0;;
esac

print_summary
(( N_FAILED > 0 )) && exit 1
exit 0