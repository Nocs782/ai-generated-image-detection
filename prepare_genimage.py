"""Prepare a fixed, balanced external test sample without training any model."""

import csv
import hashlib
import io
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent
GENERATORS = ("adm_imagenet", "biggan_imagenet", "glide_imagenet", "midjourney_imagenet",
              "sdv4_imagenet", "sdv5_imagenet", "vqdm_imagenet", "wukong_imagenet")
LABELS = {"nature": 0, "ai": 1}


def pixel_digest(image):
    return hashlib.sha256(f"RGB:{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def prepare_archive(archive, destination, manifest, audit_path, cifake_hashes, per_group=625, seed=42):
    if per_group < 1:
        raise ValueError("Each generator/class needs at least one image.")
    staging = destination.with_name(destination.name + ".partial")
    if any(path.exists() for path in (destination, staging)):
        raise FileExistsError("External data already exist; refusing to overwrite them.")
    if manifest.exists() != audit_path.exists():
        raise FileExistsError("External manifest and audit must either both exist or both be absent.")
    saved_manifest = manifest.read_text() if manifest.exists() else None
    saved_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    with ZipFile(archive) as z:
        files = [item for item in z.infolist() if not item.is_dir()]
        if len(files) != len({item.filename for item in files}):
            raise ValueError("Archive contains repeated file paths.")
        groups = defaultdict(list)
        for item in files:
            match = re.fullmatch(r"test/([a-z0-9_]+)/(ai|nature)/[^/\\]+\.(?i:png|jpe?g)", item.filename)
            if not match or match[1] not in GENERATORS:
                raise ValueError(f"Unexpected or unsafe archive path: {item.filename}")
            groups[(match[1], LABELS[match[2]])].append(item.filename)
        expected = {(generator, label) for generator in GENERATORS for label in (0, 1)}
        if set(groups) != expected or any(len(group) < per_group for group in groups.values()):
            raise ValueError("Archive needs enough images in both classes of all eight generators.")

        rng = random.Random(seed)
        rows, seen = [], {}
        duplicates = overlap = 0
        sizes, formats = Counter(), Counter()
        staging.mkdir(parents=True)
        try:
            for (generator, label), names in sorted(groups.items()):
                names = sorted(names)
                rng.shuffle(names)
                selected = 0
                for name in names:
                    raw = z.read(name)  # Reading also verifies this member's ZIP CRC.
                    with Image.open(io.BytesIO(raw)) as source:
                        image = source.convert("RGB")
                        digest = pixel_digest(image)
                        resized_digest = pixel_digest(image.resize((32, 32), Image.Resampling.BICUBIC))
                        size, image_format = f"{image.width}x{image.height}", source.format
                    if digest in cifake_hashes or resized_digest in cifake_hashes:
                        overlap += 1
                        continue
                    if digest in seen:
                        if seen[digest] != label:
                            raise ValueError(f"Identical external images have conflicting labels: {name}")
                        duplicates += 1
                        continue
                    seen[digest] = label
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    rows.append({"path": name, "label": label, "generator": generator, "digest": digest})
                    sizes[size] += 1
                    formats[f"{label}:{image_format}"] += 1
                    selected += 1
                    if selected == per_group:
                        break
                if selected != per_group:
                    raise ValueError(f"Not enough unique non-overlapping images for {generator}, class {label}.")
                print(f"Selected {selected} images: {generator}, label {label}", flush=True)

            rows.sort(key=lambda row: (row["generator"], row["label"], row["path"]))
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=("path", "label", "generator", "digest"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            contents = buffer.getvalue()
            with archive.open("rb") as handle:
                archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            audit = {
                "dataset": "GenImage external test sample", "seed": seed,
                "source": "https://huggingface.co/datasets/jzousz/GenImage/tree/main",
                "original_project": "https://github.com/GenImage-Dataset/GenImage",
                "archive": archive.name, "archive_sha256": archive_hash,
                "manifest_sha256": hashlib.sha256(contents.encode()).hexdigest(),
                "archive_images": len(files), "selected_images": len(rows),
                "images_per_class_per_generator": per_group, "labels": {"REAL": 0, "FAKE": 1},
                "selected_counts": {generator: {str(label): per_group for label in (0, 1)} for generator in GENERATORS},
                "candidate_counts": {generator: {str(label): len(groups[(generator, label)]) for label in (0, 1)}
                                     for generator in GENERATORS},
                "duplicate_candidates_skipped": duplicates, "cifake_overlap_candidates_skipped": overlap,
                "cifake_reference_unique_images": len(cifake_hashes),
                "deduplication": "SHA256 of dimensions and decoded RGB pixels across the selected sample",
                "cifake_overlap_check": "Native RGB and bicubic 32x32 RGB compared with all original CIFAKE images",
                "native_sizes": dict(sizes), "formats_by_label": dict(formats),
                "limitations": ["Unselected archive images are not decoded or CRC-checked.",
                                "Near-duplicate and pretrained-ImageNet exposure are not ruled out.",
                                "Real JPEG and generated PNG encoding differs; resizing does not remove compression traces."],
            }
            if saved_manifest is not None and (contents != saved_manifest or audit != saved_audit):
                raise ValueError("Regenerated external sample differs from the saved manifest/audit.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            if saved_manifest is None:
                with manifest.open("x", newline="") as handle:
                    handle.write(contents)
                with audit_path.open("x") as handle:
                    handle.write(json.dumps(audit, indent=2) + "\n")
            return audit
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main():
    hashes = set()
    with ZipFile(ROOT / "archive.zip") as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            with Image.open(io.BytesIO(z.read(item))) as image:
                hashes.add(pixel_digest(image.convert("RGB")))
    print(f"Loaded {len(hashes):,} unique CIFAKE image hashes for overlap checks.", flush=True)
    audit = prepare_archive(ROOT / "genimage_test.zip", ROOT / "data/genimage-test",
                            ROOT / "splits/genimage.csv", ROOT / "splits/genimage_audit.json", hashes)
    print(f"Prepared {audit['selected_images']:,} external test images; no models trained.")


if __name__ == "__main__":
    main()
