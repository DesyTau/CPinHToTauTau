#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


# =============================================================================
# Failure classification
# =============================================================================

FAILURE_PATTERNS = [
    (
        "wall time exceeded",
        re.compile(
            r"SYSTEM_PERIODIC_REMOVE.*wall time exceeded",
            re.IGNORECASE,
        ),
    ),
    (
        "memory exceeded",
        re.compile(
            r"(memory.*exceed|out of memory|oom-kill|oom_kill|MemoryError)",
            re.IGNORECASE,
        ),
    ),
    (
        "disk exceeded",
        re.compile(
            r"(disk.*exceed|no space left on device)",
            re.IGNORECASE,
        ),
    ),
    (
        "segmentation fault",
        re.compile(
            r"(segmentation fault|segfault|core dumped)",
            re.IGNORECASE,
        ),
    ),
    (
        "bootstrap failure",
        re.compile(
            r"(bootstrap file failed|bootstrap.*failed)",
            re.IGNORECASE,
        ),
    ),
    (
        "sandbox failure",
        re.compile(
            r"(sandbox.*failed|env loading failed)",
            re.IGNORECASE,
        ),
    ),
    (
        "missing sandbox bundle",
        re.compile(
            r"law_wlcg_get_file: could not determine file to load",
            re.IGNORECASE,
        ),
    ),
    (
        "file transfer failure",
        re.compile(
            r"(xrdcp.*failed|copy.*failed|transfer.*failed)",
            re.IGNORECASE,
        ),
    ),
    (
        "python traceback",
        re.compile(
            r"Traceback \(most recent call last\)",
            re.IGNORECASE,
        ),
    ),
    (
        "killed",
        re.compile(
            r"(killed by signal|signal 9|\bkilled\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "job removed",
        re.compile(
            r"job removed",
            re.IGNORECASE,
        ),
    ),
]


# =============================================================================
# Explicit successful-job patterns
# =============================================================================

SUCCESS_PATTERNS = [
    re.compile(
        r"job exit code\s*[:=]?\s*0\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"task exit code\s*[:=]?\s*0\b",
        re.IGNORECASE,
    ),
]


# =============================================================================
# Helpers
# =============================================================================

def read_file(path: Path) -> str:

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception as exc:
        return f"ERROR_READING_FILE: {exc}"


def classify_failure(text: str) -> list[str]:
    """
    Return all matching failure classes.
    """

    reasons = []

    for name, pattern in FAILURE_PATTERNS:
        if pattern.search(text):
            reasons.append(name)

    return reasons


def find_failure_excerpt(
    text: str,
    reasons: list[str],
    context: int = 1,
) -> str:
    """
    Find a useful line around the first matched failure.
    """

    lines = text.splitlines()

    for name, pattern in FAILURE_PATTERNS:

        if name not in reasons:
            continue

        for i, line in enumerate(lines):

            if pattern.search(line):

                start = max(
                    0,
                    i - context,
                )

                stop = min(
                    len(lines),
                    i + context + 1,
                )

                excerpt = " | ".join(
                    line.strip()
                    for line in lines[start:stop]
                    if line.strip()
                )

                return excerpt

    return ""


def find_exit_codes(text: str) -> list[int]:
    """
    Extract explicit exit codes found in the log.
    """

    patterns = [
        r"job exit code\s*[:=]?\s*(-?\d+)",
        r"task exit code\s*[:=]?\s*(-?\d+)",
        r"exit code\s*[:=]?\s*(-?\d+)",
        r"exit_code\s*[:=]?\s*(-?\d+)",
    ]

    codes = []

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):

            code = int(
                match.group(1)
            )

            if code not in codes:
                codes.append(code)

    return codes


def get_job_name(path: Path) -> str:
    """
    Return the filename without extension.
    """

    return path.stem


def get_task_from_path(
    path: Path,
    root: Path,
) -> str:
    """
    Extract the ColumnFlow task name from the path.
    """

    if root.name.startswith("cf."):
        return root.name

    relative = path.relative_to(
        root
    )

    if relative.parts:
        return relative.parts[0]

    return "unknown"


def get_path_info(
    path: Path,
    root: Path,
    version: str | None,
) -> dict[str, str]:
    """
    Extract useful information from the ColumnFlow output path.

    Typical structure:

        cf.TASK/
        CONFIG/
        DATASET/
        SHIFT/
        PARAMETERS/
        VERSION/
        stdout_*.txt

    Different ColumnFlow tasks can have different numbers of parameter
    directories. Therefore only task/config/dataset are assumed to have
    relatively stable positions. Everything between dataset and version
    is retained in "parameters".
    """

    relative = path.relative_to(
        root
    )

    parts = list(
        relative.parts
    )

    if root.name.startswith("cf."):

        task = root.name
        inner_parts = parts

    else:

        if not parts:
            return {
                "task": "unknown",
                "config": "unknown",
                "dataset": "unknown",
                "shift": "unknown",
                "parameters": "",
                "relative_path": str(relative),
            }

        task = parts[0]
        inner_parts = parts[1:]


    # Remove filename.
    directory_parts = inner_parts[:-1]


    # Restrict path information to everything before the requested version.
    if version and version in directory_parts:

        version_index = directory_parts.index(
            version
        )

        before_version = directory_parts[
            :version_index
        ]

    else:

        before_version = directory_parts


    config = (
        before_version[0]
        if len(before_version) >= 1
        else "unknown"
    )

    dataset = (
        before_version[1]
        if len(before_version) >= 2
        else "unknown"
    )


    # Try to identify the shift.
    #
    # Most task paths put the shift directly after dataset. However,
    # not every task necessarily follows the same layout, so only
    # recognize common shift names here.

    shift = "unknown"

    if len(before_version) >= 3:

        candidate = before_version[2]

        if (
            candidate == "nominal"
            or candidate.endswith("_up")
            or candidate.endswith("_down")
            or "__up" in candidate
            or "__down" in candidate
        ):
            shift = candidate


    # Keep all task-specific directories. This makes the script robust
    # against CalibrateEvents, ReduceEvents, ProduceColumns, merge tasks,
    # etc. having different layouts.

    parameters = ""

    if len(before_version) >= 3:

        parameters = "/".join(
            before_version[2:]
        )


    return {
        "task": task,
        "config": config,
        "dataset": dataset,
        "shift": shift,
        "parameters": parameters,
        "relative_path": str(relative),
    }


def find_task_dirs(
    root: Path,
    requested_tasks: list[str] | None,
) -> list[Path]:
    """
    Find cf.* directories directly below the analysis root.

    If root itself is a cf.* directory, inspect only that directory.
    """

    if root.name.startswith("cf."):

        task_dirs = [
            root
        ]

    else:

        task_dirs = sorted(
            path
            for path in root.iterdir()
            if (
                path.is_dir()
                and path.name.startswith("cf.")
            )
        )


    if requested_tasks:

        requested = {
            task
            if task.startswith("cf.")
            else f"cf.{task}"
            for task in requested_tasks
        }

        task_dirs = [
            path
            for path in task_dirs
            if path.name in requested
        ]


    return task_dirs


def find_log_files(
    task_dirs: list[Path],
    version: str | None,
    shift: str | None,
) -> list[Path]:
    """
    Recursively find log txt files in all requested ColumnFlow tasks.
    """

    log_files = []

    for task_dir in task_dirs:

        for path in task_dir.rglob("*.txt"):

            if not path.is_file():
                continue

            if (
                version is not None
                and version not in path.parts
            ):
                continue

            if (
                shift is not None
                and shift not in path.parts
            ):
                continue

            log_files.append(
                path
            )

    return sorted(
        log_files
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect ColumnFlow HTCondor logs and summarize failed jobs."
        )
    )


    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Path to the ColumnFlow analysis store, e.g. "
            ".../data/analysis_MSSM_H_tt_skim_2025_v1"
        ),
    )


    parser.add_argument(
        "--version",
        default="all_mass_bdt_test_task_scheduling",
        help=(
            "production version; use --version all to inspect every version"
        ),
    )


    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help=(
            "restrict to a ColumnFlow task. Can be given multiple times, "
            "e.g. --task CalibrateEvents --task ReduceEvents. "
            "Default: inspect all cf.* directories"
        ),
    )


    parser.add_argument(
        "--shift",
        default=None,
        help=(
            "restrict to a single shift, e.g. nominal; "
            "default: inspect all shifts"
        ),
    )


    parser.add_argument(
        "--csv",
        default="failed_jobs.csv",
        help="output CSV file",
    )


    args = parser.parse_args()


    root = args.root.expanduser().resolve()


    if not root.exists():

        raise FileNotFoundError(
            f"directory does not exist:\n{root}"
        )


    if not root.is_dir():

        raise NotADirectoryError(
            f"path is not a directory:\n{root}"
        )


    version = (
        None
        if args.version.lower() == "all"
        else args.version
    )


    # =========================================================================
    # Find task directories
    # =========================================================================

    task_dirs = find_task_dirs(
        root,
        args.task,
    )


    if not task_dirs:

        print()
        print(
            f"No cf.* task directories found below:\n{root}"
        )
        print()

        return


    # =========================================================================
    # Find logs
    # =========================================================================

    log_files = find_log_files(
        task_dirs,
        version,
        args.shift,
    )


    print()
    print("============================================================")
    print("ColumnFlow job failure check")
    print("============================================================")
    print(f"Root:       {root}")

    if version:
        print(f"Version:    {version}")
    else:
        print("Version:    all")

    if args.shift:
        print(f"Shift:      {args.shift}")
    else:
        print("Shift:      all")

    print()
    print("Tasks:")

    for task_dir in task_dirs:
        print(
            f"  {task_dir.name}"
        )

    print()
    print(f"Log files:  {len(log_files)}")
    print("============================================================")
    print()


    if not log_files:

        print(
            "No log files found."
        )

        print()
        print(
            "If you want to inspect every production version, run with:"
        )
        print()
        print(
            "  --version all"
        )
        print()

        return


    # =========================================================================
    # Analyze logs
    # =========================================================================

    failures = []
    successful = []
    unknown = []


    for log_file in log_files:

        text = read_file(
            log_file
        )


        info = get_path_info(
            log_file,
            root,
            version,
        )


        task = info[
            "task"
        ]

        config = info[
            "config"
        ]

        dataset = info[
            "dataset"
        ]

        shift = info[
            "shift"
        ]

        parameters = info[
            "parameters"
        ]

        relative_path = info[
            "relative_path"
        ]


        job = get_job_name(
            log_file
        )


        reasons = classify_failure(
            text
        )


        exit_codes = find_exit_codes(
            text
        )


        # Explicit non-zero exit code is always a failure.

        nonzero_codes = [
            code
            for code in exit_codes
            if code != 0
        ]


        if nonzero_codes:

            reasons.append(
                "non-zero exit code"
            )


        # Remove duplicates while preserving order.

        reasons = list(
            dict.fromkeys(
                reasons
            )
        )


        if reasons:

            failures.append({
                "task": task,
                "config": config,
                "dataset": dataset,
                "shift": shift,
                "parameters": parameters,
                "job": job,
                "reason": "; ".join(
                    reasons
                ),
                "exit_codes": ",".join(
                    map(
                        str,
                        exit_codes,
                    )
                ),
                "excerpt": find_failure_excerpt(
                    text,
                    reasons,
                ),
                "relative_path": relative_path,
                "file": str(
                    log_file
                ),
            })

            continue


        # Determine whether there is an explicit success marker.

        is_success = any(
            pattern.search(
                text
            )
            for pattern in SUCCESS_PATTERNS
        )


        if is_success:

            successful.append({
                "task": task,
                "file": log_file,
            })

        else:

            unknown.append({
                "task": task,
                "file": log_file,
            })


    # =========================================================================
    # Summary
    # =========================================================================

    print("SUMMARY")
    print("-------")
    print(f"Total log files : {len(log_files)}")
    print(f"Failed          : {len(failures)}")
    print(f"Successful      : {len(successful)}")
    print(f"Unknown         : {len(unknown)}")
    print()


    # =========================================================================
    # Per-task summary
    # =========================================================================

    all_task_counter = Counter(
        get_task_from_path(
            path,
            root,
        )
        for path in log_files
    )


    failed_task_counter = Counter(
        failure["task"]
        for failure in failures
    )


    successful_task_counter = Counter(
        entry["task"]
        for entry in successful
    )


    unknown_task_counter = Counter(
        entry["task"]
        for entry in unknown
    )


    print("STATUS PER TASK")
    print("---------------")

    print(
        f"{'task':35s}"
        f"{'total':>10s}"
        f"{'failed':>10s}"
        f"{'success':>10s}"
        f"{'unknown':>10s}"
    )

    print(
        "-" * 75
    )


    for task in sorted(
        all_task_counter
    ):

        print(
            f"{task:35s}"
            f"{all_task_counter[task]:10d}"
            f"{failed_task_counter[task]:10d}"
            f"{successful_task_counter[task]:10d}"
            f"{unknown_task_counter[task]:10d}"
        )

    print()


    # =========================================================================
    # Failure reasons
    # =========================================================================

    reason_counter = Counter()


    for failure in failures:

        for reason in failure[
            "reason"
        ].split("; "):

            reason_counter[
                reason
            ] += 1


    if failures:

        print("FAILURE REASONS")
        print("---------------")

        for reason, count in reason_counter.most_common():

            print(
                f"{count:6d}  {reason}"
            )

        print()


    # =========================================================================
    # Failures per task
    # =========================================================================

    if failed_task_counter:

        print("FAILED JOBS PER TASK")
        print("--------------------")

        for task, count in failed_task_counter.most_common():

            print(
                f"{count:6d}  {task}"
            )

        print()


    # =========================================================================
    # Failures per dataset
    # =========================================================================

    dataset_counter = Counter(
        failure["dataset"]
        for failure in failures
        if failure["dataset"] != "unknown"
    )


    if dataset_counter:

        print("FAILED JOBS PER DATASET")
        print("-----------------------")

        for dataset, count in dataset_counter.most_common():

            print(
                f"{count:6d}  {dataset}"
            )

        print()


    # =========================================================================
    # Full failed-job list
    # =========================================================================

    if failures:

        print("FAILED JOBS")
        print("-----------")


        for i, failure in enumerate(
            failures,
            start=1,
        ):

            print(
                f"[{i}/{len(failures)}]"
            )

            print(
                f"  task    : {failure['task']}"
            )

            if failure["config"] != "unknown":

                print(
                    f"  config  : {failure['config']}"
                )

            if failure["dataset"] != "unknown":

                print(
                    f"  dataset : {failure['dataset']}"
                )

            if failure["shift"] != "unknown":

                print(
                    f"  shift   : {failure['shift']}"
                )

            if failure["parameters"]:

                print(
                    f"  params  : {failure['parameters']}"
                )

            print(
                f"  job     : {failure['job']}"
            )

            print(
                f"  reason  : {failure['reason']}"
            )

            if failure["exit_codes"]:

                print(
                    f"  exit    : {failure['exit_codes']}"
                )

            if failure["excerpt"]:

                print(
                    f"  message : {failure['excerpt']}"
                )

            print(
                f"  log     : {failure['file']}"
            )

            print()


    # =========================================================================
    # Unknown logs
    # =========================================================================

    if unknown:

        print("UNKNOWN JOB STATUS")
        print("------------------")

        print(
            "These logs contain neither a recognized failure nor "
            "an explicit successful exit code:"
        )

        for entry in unknown:

            print(
                f"  [{entry['task']}] {entry['file']}"
            )

        print()


    # =========================================================================
    # CSV output
    # =========================================================================

    csv_path = Path(
        args.csv
    )


    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "task",
            "config",
            "dataset",
            "shift",
            "parameters",
            "job",
            "reason",
            "exit_codes",
            "excerpt",
            "relative_path",
            "file",
        ]


        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        writer.writerows(
            failures
        )


    print(
        f"Failure report written to: {csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()