#!/usr/bin/env python3

import os
import shutil

BASE_DIR = (
    "/eos/project/d/desytau/public/jmalvaso/MSSM_H_tt_store/"
    "analysis_MSSM_H_tt_skim_2025_v1"
)

SEARCH_DIRS = [
    "cf.ReduceEvents",
]

TARGET_NAME = "all_mass_bdt_test_1"

# True = only show what would be removed
# False = actually remove the directories
DRY_RUN = False


def main():
    found = []

    for search_dir in SEARCH_DIRS:
        base = os.path.join(BASE_DIR, search_dir)

        if not os.path.isdir(base):
            print(f"Directory not found: {base}")
            continue

        for root, dirs, files in os.walk(base, topdown=False):
            for dirname in dirs:
                if dirname != TARGET_NAME:
                    continue

                path = os.path.join(root, dirname)

                if os.path.islink(path):
                    print(f"Skipping symlink: {path}")
                    continue

                found.append(path)

    if not found:
        print(f'No directories named "{TARGET_NAME}" found.')
        return

    print(f'Found {len(found)} directories named "{TARGET_NAME}":\n')

    for path in sorted(found):
        print(path)

    if DRY_RUN:
        print(
            "\nDRY RUN: nothing was removed.\n"
            "Set DRY_RUN = False to actually delete them."
        )
        return

    print("\nRemoving directories...\n")

    removed = 0

    for path in found:
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
            removed += 1
        except Exception as exc:
            print(f"ERROR removing {path}: {exc}")

    print(f"\nDone. Removed {removed}/{len(found)} directories.")


if __name__ == "__main__":
    main()