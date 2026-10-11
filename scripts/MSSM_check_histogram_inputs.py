#!/usr/bin/env python3
# coding: utf-8

"""Check downstream dependencies without running tasks or submitting jobs."""

import argparse
import sys


VERSION = "all_mass_bdt_compact_v1"


def make_roots(args, task_classes):
    """Check the exact stage being submitted, including each column producer."""
    kwargs = dict(
        version=VERSION, config=args.config, shift="nominal",
        calibrators=tuple(filter(None, args.calibrators.split(","))),
        selector=args.selector, workflow="local", pilot=True,
    )
    producers = tuple(filter(None, args.producers.split(",")))
    task_cls = task_classes[args.stage]
    for dataset in filter(None, args.datasets.split(",")):
        if args.stage == "produce":
            for producer in producers:
                yield task_cls(dataset=dataset, producer=producer, **kwargs)
        else:
            hist_kwargs = {"hist_producer": args.hist_producer} if args.hist_producer else {}
            masses = getattr(args, "histogram_bdt_masses", "")
            if masses:
                hist_kwargs["histogram_bdt_masses"] = tuple(filter(None, masses.split(",")))
            yield task_cls(
                dataset=dataset, producers=producers,
                variables=tuple(filter(None, args.variables.split(","))),
                **kwargs, **hist_kwargs,
            )


def check_dependencies(root, protected_classes):
    """Stop at completed tasks; refuse incomplete event-processing tasks."""
    seen = set()
    pending = [root]
    failures = []
    while pending:
        task = pending.pop()
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        try:
            complete = task.complete()
            if isinstance(task, protected_classes):
                if not complete:
                    failures.append(f"missing existing output: {task}")
                if task.version != VERSION:
                    failures.append(f"unexpected upstream version: {task}")
                continue
            if not complete:
                pending.extend(task.deps())
        except Exception as exc:
            failures.append(f"cannot verify {task.task_family}: {exc}")
    return seen, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True, help="comma-separated dataset names")
    parser.add_argument("--variables", default="bdt_D_DY_M100")
    parser.add_argument("--shifts", default="nominal", help="comma-separated names or patterns")
    parser.add_argument("--hist-producer", default=None, help="histogram producer name; default: config setting")
    parser.add_argument("--stage", choices=("produce", "create-hists", "merge-hists"), default="merge-hists")
    parser.add_argument("--producers", default="main")
    parser.add_argument("--calibrators", default="main")
    parser.add_argument("--selector", default="main")
    parser.add_argument("--version", default=VERSION, choices=(VERSION,))
    parser.add_argument("--histogram-bdt-masses",default="",help="ordinary histogram BDT masses: 100, 100,200, or all")
    args = parser.parse_args()

    import law
    # Standalone Python has no LAW command-line parser. Some LAW releases
    # return None here, while ColumnFlow's req_params expects a mapping.
    cli_values = law.parser.global_cmdline_values
    law.parser.global_cmdline_values = lambda *a, **kw: cli_values(*a, **kw) or {}
    from columnflow.tasks.calibration import CalibrateEvents
    from columnflow.tasks.selection import SelectEvents
    from columnflow.tasks.reduction import ReduceEvents, MergeReducedEvents, ProvideReducedEvents
    from columnflow.tasks.histograms import CreateHistograms, MergeHistograms
    from columnflow.tasks.production import ProduceColumns

    # Match MSSM_submit_stage.sh: always consume the existing merged events.
    ProvideReducedEvents.force_merging = ProvideReducedEvents.force_merging.copy(default=True)
    protected = (CalibrateEvents, SelectEvents, ReduceEvents, MergeReducedEvents)
    patterns = tuple(filter(None, args.shifts.split(",")))
    all_failures = []
    task_classes = {"produce": ProduceColumns, "create-hists": CreateHistograms, "merge-hists": MergeHistograms}
    roots = list(make_roots(args, task_classes))
    if not roots:
        print("No tasks selected: check datasets and producers.", file=sys.stderr)
        return 1
    for nominal in roots:
        dataset = nominal.dataset
        shifts = [
            shift.name for shift in nominal.config_inst.shifts
            if law.util.multi_match(shift.name, patterns, mode=any)
            and (not nominal.dataset_inst.is_data or shift.name == "nominal")
        ]
        if not shifts:
            all_failures.append(f"no shifts matched for dataset {dataset}")
        for shift in shifts:
            task = type(nominal).req(nominal, shift=shift)
            seen, failures = check_dependencies(task, protected)
            print(f"{dataset} / {shift}: checked {len(seen)} tasks; {len(failures)} problems")
            all_failures.extend(failures)

    if all_failures:
        print("\nDo not submit: existing event outputs or required metadata are unavailable.", file=sys.stderr)
        for failure in dict.fromkeys(all_failures):
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nInputs ready under {VERSION}. This check did not run or submit any task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())