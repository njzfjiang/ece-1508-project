"""Package a deterministic reduced DarkDriving dataset for Colab smoke tests."""

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_img2img_turbo_data import load_training_pairs, matching_images


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_pair(pair: tuple[Path, Path], day_dir: Path, night_dir: Path) -> None:
    day_source, night_source = pair
    day_dir.mkdir(parents=True, exist_ok=True)
    night_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(day_source, day_dir / day_source.name)
    shutil.copy2(night_source, night_dir / night_source.name)


def package_smoke_dataset(
    processed_root: Path,
    output: Path,
    shot: int = 10,
    seed: int = 1,
    test_pairs: int = 50,
    test_seed: int = 1508,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """Create a tar.gz containing the canonical processed-data hierarchy."""
    processed_root = processed_root.resolve()
    output = output.resolve()

    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Output filename must end with .tar.gz")
    if shot < 1 or seed < 1:
        raise ValueError("shot and seed must be positive integers")
    if test_pairs < 1:
        raise ValueError("test_pairs must be a positive integer")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")

    train_pairs = load_training_pairs(processed_root, shot, seed)
    all_test_pairs = matching_images(
        processed_root / "day2night" / "test" / "day",
        processed_root / "day2night" / "test" / "night",
    )
    if test_pairs > len(all_test_pairs):
        raise ValueError(
            f"Requested {test_pairs} test pairs, but only {len(all_test_pairs)} exist"
        )

    sampled_indices = sorted(
        random.Random(test_seed).sample(range(len(all_test_pairs)), test_pairs)
    )
    selected_test_pairs = [all_test_pairs[index] for index in sampled_indices]
    split_source = (
        processed_root
        / "splits"
        / "fewshot"
        / f"{shot}shot"
        / f"seed{seed}"
        / "split.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="darkdriving-smoke-") as temporary:
        staging_root = Path(temporary)
        archive_processed = staging_root / "data" / "processed"

        for pair in train_pairs:
            copy_pair(
                pair,
                archive_processed / "day2night" / "train" / "day",
                archive_processed / "day2night" / "train" / "night",
            )
        for pair in selected_test_pairs:
            copy_pair(
                pair,
                archive_processed / "day2night" / "test" / "day",
                archive_processed / "day2night" / "test" / "night",
            )

        split_destination = (
            archive_processed
            / "splits"
            / "fewshot"
            / f"{shot}shot"
            / f"seed{seed}"
            / "split.json"
        )
        split_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_source, split_destination)

        payload_bytes = sum(
            path.stat().st_size for path in staging_root.rglob("*") if path.is_file()
        )
        manifest = {
            "dataset": "DarkDriving",
            "purpose": "Colab smoke test only",
            "shot": shot,
            "seed": seed,
            "train_pairs": len(train_pairs),
            "test_pairs": len(selected_test_pairs),
            "test_seed": test_seed,
            "test_filenames": [day.name for day, _ in selected_test_pairs],
            "payload_bytes": payload_bytes,
            "archive_layout": "data/processed",
        }
        manifest_path = archive_processed / "smoke_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        temporary_output = staging_root / output.name
        with tarfile.open(temporary_output, "w:gz") as archive:
            archive.add(
                staging_root / "data",
                arcname="data",
                recursive=True,
            )
        if output.exists():
            output.unlink()
        shutil.move(str(temporary_output), output)

    checksum = sha256_file(output)
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    manifest.update(
        {
            "archive": output.name,
            "archive_bytes": output.stat().st_size,
            "sha256": checksum,
            "checksum_file": checksum_path.name,
        }
    )
    return output, manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a reduced DarkDriving archive for Colab smoke tests"
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/darkdriving_smoke.tar.gz"),
    )
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-pairs", type=int, default=50)
    parser.add_argument("--test-seed", type=int, default=1508)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output, manifest = package_smoke_dataset(
        processed_root=args.processed_root,
        output=args.output,
        shot=args.shot,
        seed=args.seed,
        test_pairs=args.test_pairs,
        test_seed=args.test_seed,
        overwrite=args.overwrite,
    )
    print(f"Created: {output}")
    print(f"SHA256: {manifest['sha256']}")
    print(
        f"Contents: {manifest['train_pairs']} train pairs, "
        f"{manifest['test_pairs']} test pairs"
    )
    print(f"Archive size: {manifest['archive_bytes'] / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
