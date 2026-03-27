import shutil
import argparse
from pathlib import Path

# Deletes saved false positive / false negative composite images to free up disk space.
# The pairs_list.jsonl logs are kept so the error pair indices are not lost.
#
# Usage:
#   python scripts/clean_error_outputs.py              # delete all composite images
#   python scripts/clean_error_outputs.py --dry-run    # preview without deleting

ROOT = Path(__file__).resolve().parent.parent
FP_DIR = ROOT / "outputs" / "false_positives"
FN_DIR = ROOT / "outputs" / "false_negatives"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without actually deleting anything."
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run - nothing will be deleted.\n")

    deleted = 0
    for base_dir in [FP_DIR, FN_DIR]:
        if not base_dir.exists():
            continue
        # each run gets its own subdirectory; delete only the composites/ folder inside each
        for run_dir in sorted(base_dir.iterdir()):
            composites = run_dir / "composites"
            if composites.exists():
                if args.dry_run:
                    count = len(list(composites.glob("*.jpg")))
                    print(f"  would delete: {composites}  ({count} images)")
                else:
                    count = len(list(composites.glob("*.jpg")))
                    shutil.rmtree(composites)
                    print(f"  deleted: {composites}  ({count} images)")
                deleted += count

    if not args.dry_run:
        print(f"\nDone. {deleted} images removed. pairs_list.jsonl logs kept.")
    else:
        print(f"\n{deleted} images would be removed.")


if __name__ == "__main__":
    main()
