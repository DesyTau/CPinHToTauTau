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


def get_dataset_from_path(
    path: Path,
    root: Path,
) -> str:

    relative = path.relative_to(
        root
    )

    # Expected:
    #
    # DATASET/
    # nominal/
    # calib__main/
    # VERSION/
    # stdout_*.txt

    return relative.parts[0]


def get_shift_from_path(
    path: Path,
    root: Path,
) -> str:

    relative = path.relative_to(
        root
    )

    if len(relative.parts) >= 2:
        return relative.parts[1]

    return "unknown"


def get_job_name(path: Path) -> str:
    return path.stem


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
            "Path to the config directory below cf.CalibrateEvents, "
            "e.g. .../cf.CalibrateEvents/run3_2022_postEE_emu"
        ),
    )

    parser.add_argument(
        "--version",
        default="all_mass_bdt_test_task_scheduling",
        help="production version",
    )

    parser.add_argument(
        "--calibrator",
        default="calib__main",
        help="calibrator directory",
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


    # =========================================================================
    # Find logs
    # =========================================================================

    if args.shift:

        pattern = (
            f"*/{args.shift}/"
            f"{args.calibrator}/"
            f"{args.version}/*.txt"
        )

    else:

        pattern = (
            f"*/*/"
            f"{args.calibrator}/"
            f"{args.version}/*.txt"
        )


    log_files = sorted(
        root.glob(pattern)
    )

    print()
    print("============================================================")
    print("ColumnFlow job failure check")
    print("============================================================")
    print(f"Root:       {root}")
    print(f"Version:    {args.version}")
    print(f"Calibrator: {args.calibrator}")

    if args.shift:
        print(f"Shift:      {args.shift}")
    else:
        print("Shift:      all")

    print(f"Log files:  {len(log_files)}")
    print("============================================================")
    print()


    if not log_files:
        print(
            "No log files found."
        )
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

        dataset = get_dataset_from_path(
            log_file,
            root,
        )

        shift = get_shift_from_path(
            log_file,
            root,
        )

        job = get_job_name(
            log_file
        )

        reasons = classify_failure(
            text
        )

        exit_codes = find_exit_codes(
            text
        )

        # Explicit non-zero exit code is also a failure.
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
            dict.fromkeys(reasons)
        )

        if reasons:

            failures.append({
                "dataset": dataset,
                "shift": shift,
                "job": job,
                "file": str(log_file),
                "reason": "; ".join(reasons),
                "exit_codes": ",".join(
                    map(str, exit_codes)
                ),
                "excerpt": find_failure_excerpt(
                    text,
                    reasons,
                ),
            })

            continue


        # Determine whether there is an explicit success marker.
        is_success = any(
            pattern.search(text)
            for pattern in SUCCESS_PATTERNS
        )

        if is_success:
            successful.append(
                log_file
            )
        else:
            unknown.append(
                log_file
            )


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
    # Failure reasons
    # =========================================================================

    reason_counter = Counter()

    for failure in failures:
        for reason in failure["reason"].split("; "):
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
    # Failures per dataset
    # =========================================================================

    dataset_counter = Counter(
        failure["dataset"]
        for failure in failures
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
                f"  dataset : {failure['dataset']}"
            )

            print(
                f"  shift   : {failure['shift']}"
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

        for path in unknown:
            print(
                f"  {path}"
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
            "dataset",
            "shift",
            "job",
            "reason",
            "exit_codes",
            "excerpt",
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