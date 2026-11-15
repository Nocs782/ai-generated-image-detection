"""Validate/extract CIFAKE and write a fixed split without identical-pixel leakage."""

import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent
SEED = 42
LABELS = {"REAL": 0, "FAKE": 1}


def make_splits(records):
    test = [row for row in records if row["path"].startswith("test/")]
    test_hashes = {row["digest"] for row in test}
    pool = []
    seen = set()
    excluded_overlap = excluded_duplicates = 0
    for row in sorted(records, key=lambda row: row["path"]):
        if row["path"].startswith("test/"):
            continue
        if row["digest"] in test_hashes:
            excluded_overlap += 1
        elif row["digest"] in seen:
            excluded_duplicates += 1
        else:
            pool.append(row)
            seen.add(row["digest"])

    result = [dict(row, split="test") for row in test]
    rng = random.Random(SEED)
    for label in LABELS.values():
        group = [row for row in pool if row["label"] == label]
        rng.shuffle(group)
        validation_count = round(len(group) * 0.2)
        if validation_count < 1 or validation_count >= len(group):
            raise ValueError("Each class needs enough unique images for train and validation.")
        for index, row in enumerate(group):
            result.append(dict(row, split="val" if index < validation_count else "train"))

    return sorted(result, key=lambda row: (row["split"], row["path"])), {
        "training_images_excluded_for_test_overlap": excluded_overlap,
        "additional_training_duplicates_excluded": excluded_duplicates,
        "duplicates_within_official_test_retained": len(test) - len(test_hashes),
    }


def main():
    archive = ROOT / "archive.zip"
    destination = ROOT / "data/cifake"
    staging = ROOT / "data/cifake.partial"
    manifest = ROOT / "splits/cifake.csv"
    audit_path = ROOT / "splits/data_audit.json"
    if any(path.exists() for path in (destination, staging, manifest, audit_path)):
        raise FileExistsError("Prepared data or splits already exist; refusing to overwrite them.")

    with ZipFile(archive) as z:
        names = [item.filename for item in z.infolist() if not item.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains repeated file paths.")
        if any(not re.fullmatch(r"(train|test)/(REAL|FAKE)/[^/\\]+\.jpg", name) for name in names):
            raise ValueError("Archive contains an unexpected or unsafe file path.")
        expected = {"train/REAL": 50000, "train/FAKE": 50000, "test/REAL": 10000, "test/FAKE": 10000}
        if dict(Counter(str(Path(name).parent) for name in names)) != expected:
            raise ValueError("Archive does not have CIFAKE's expected class/split counts.")

        staging.mkdir(parents=True)
        records = []
        digest_labels = {}
        try:
            for index, name in enumerate(sorted(names), 1):
                raw = z.read(name)
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.size != (32, 32) or image.mode != "RGB":
                        raise ValueError(f"Expected 32x32 RGB image: {name}")
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                label = LABELS[Path(name).parts[1]]
                if digest in digest_labels and digest_labels[digest] != label:
                    raise ValueError(f"Identical images have conflicting labels: {name}")
                digest_labels[digest] = label
                records.append({"path": name, "label": label, "digest": digest})
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                if index % 30000 == 0:
                    print(f"Validated and extracted {index:,}/{len(names):,} images", flush=True)

            rows, audit = make_splits(records)
            by_split = {split: {str(label): sum(row["split"] == split and row["label"] == label for row in rows)
                                for label in LABELS.values()} for split in ("train", "val", "test")}
            with archive.open("rb") as handle:
                archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            audit.update({"seed": SEED, "python": sys.version, "labels": LABELS, "split_counts": by_split,
                          "archive_sha256": archive_hash,
                          "deduplication": "SHA256 of decoded RGB pixels; official test set preserved"})
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("path", "label", "split"), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
            staging.rename(destination)
        except Exception:
            shutil.rmtree(staging)
            raise
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
