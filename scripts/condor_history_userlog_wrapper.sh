#!/bin/bash
set -euo pipefail

real_condor_history="${REAL_CONDOR_HISTORY:-condor_history}"
userlog_dir="${CF_HTCONDOR_USERLOG_DIR:-$HOME/htcondor_userlogs_columnflow}"

args=("$@")
all_args=" ${args[*]} "

cluster=""
proc=""

if [[ "$all_args" =~ ClusterId[[:space:]]*==[[:space:]]*([0-9]+) ]]; then
    cluster="${BASH_REMATCH[1]}"
fi

if [[ "$all_args" =~ ProcId[[:space:]]*==[[:space:]]*([0-9]+) ]]; then
    proc="${BASH_REMATCH[1]}"
fi

filtered_args=()
skip_next=0

for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "$skip_next" -eq 1 ]]; then
        skip_next=0
        continue
    fi

    case "${args[$i]}" in
        -search|-scanlimit|-name|-pool|-match|-limit)
            skip_next=1
            ;;
        -inherit|-stream-results)
            ;;
        *)
            filtered_args+=("${args[$i]}")
            ;;
    esac
done

find_userlog() {
    local cluster="$1"
    local log_file=""

    # Fast path for old flat layout.
    if [[ -f "${userlog_dir}/${cluster}.log" ]]; then
        echo "${userlog_dir}/${cluster}.log"
        return 0
    fi

    # Fast path for old law-style flat names, e.g. 19906680_0To1.log.
    log_file="$(find "$userlog_dir" -maxdepth 1 -type f -name "${cluster}*.log" -print -quit 2>/dev/null || true)"
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        echo "$log_file"
        return 0
    fi

    # Cache path after the first recursive lookup.
    local prefix="${cluster:0:4}"
    local index_dir="${userlog_dir}/.index/${prefix}"
    local index_file="${index_dir}/${cluster}.path"

    if [[ -f "$index_file" ]]; then
        log_file="$(cat "$index_file" 2>/dev/null || true)"
        if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo "$log_file"
            return 0
        fi
    fi

    # Recursive lookup in the partitioned layout.
    log_file="$(
        find "$userlog_dir" \
            -path "${userlog_dir}/.index" -prune -o \
            -type f \( -name "${cluster}.log" -o -name "${cluster}_*.log" -o -name "*${cluster}*.log" \) \
            -print -quit 2>/dev/null || true
    )"

    if [[ -n "$log_file" && -f "$log_file" ]]; then
        mkdir -p "$index_dir" 2>/dev/null || true
        printf '%s\n' "$log_file" > "$index_file" 2>/dev/null || true
        echo "$log_file"
        return 0
    fi

    return 1
}

if [[ -n "$cluster" ]]; then
    log_file="$(find_userlog "$cluster" || true)"

    if [[ -n "${log_file:-}" && -f "$log_file" ]]; then
        if [[ -n "$proc" ]]; then
            exec "$real_condor_history" \
                -userlog "$log_file" \
                -limit 1 \
                "${cluster}.${proc}" \
                "${filtered_args[@]}"
        else
            exec "$real_condor_history" \
                -userlog "$log_file" \
                -limit 1 \
                "$cluster" \
                "${filtered_args[@]}"
        fi
    fi
fi

# Bounded schedd-history fallback only.
if [[ -n "$cluster" ]]; then
    since=$((cluster - 10000))
    if [[ "$since" -lt 1 ]]; then
        since=1
    fi

    if [[ -n "$proc" ]]; then
        exec "$real_condor_history" \
            -limit 1 \
            -since "$since" \
            "${cluster}.${proc}" \
            "${filtered_args[@]}"
    else
        exec "$real_condor_history" \
            -limit 1 \
            -since "$since" \
            "$cluster" \
            "${filtered_args[@]}"
    fi
fi

exec "$real_condor_history" -limit 1 "${filtered_args[@]}"