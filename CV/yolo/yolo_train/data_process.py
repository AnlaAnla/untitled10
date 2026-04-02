from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_IMAGE_DIR = Path(r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project\imgs")
DEFAULT_LABEL_DIR = Path(r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project\labels")
DEFAULT_OUT_DIR = Path(r"C:\Code\ML\Image_SEGMENT\video_card_hand_project_yolo_dataset")
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
BACKGROUND_GROUP = "__background__"
SUPPORTED_SHAPES = {"polygon", "rectangle", "circle"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    json_path: Path | None
    labels: tuple[str, ...]

    @property
    def split_key(self) -> str:
        if self.labels:
            return "|".join(self.labels)
        return BACKGROUND_GROUP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe polygon annotations into a YOLO segmentation dataset.",
    )
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="Source image directory.")
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR), help="LabelMe JSON directory.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output YOLO dataset directory.")
    parser.add_argument(
        "--classes",
        type=str,
        default="card,hand",
        help='Comma-separated class names without background, or "auto" to discover from JSON.',
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Recommended range: 0.1-0.2.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help="Allowed image suffixes separated by commas.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include images without JSON labels and export them with empty YOLO label files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately if a sample cannot be converted instead of skipping it.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_text_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")


def build_image_index(image_dir: Path, image_exts: tuple[str, ...]) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue
        if path.stem in image_index:
            raise ValueError(f"Found duplicate image stem with different suffixes: {path.stem}")
        image_index[path.stem] = path
    return image_index


def build_json_index(label_dir: Path) -> dict[str, Path]:
    json_index: dict[str, Path] = {}
    for path in sorted(label_dir.rglob("*.json")):
        if path.stem in json_index:
            raise ValueError(f"Found duplicate JSON label stem: {path.stem}")
        json_index[path.stem] = path
    return json_index


def discover_labels(json_paths: list[Path]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for json_path in json_paths:
        data = load_json(json_path)
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def parse_classes(raw_classes: str, discovered_labels: list[str]) -> list[str]:
    if raw_classes.strip().lower() == "auto":
        classes = discovered_labels
    else:
        classes = [item.strip() for item in raw_classes.split(",") if item.strip()]

    if not classes:
        raise ValueError("No classes were provided.")

    deduped: list[str] = []
    seen: set[str] = set()
    for label in classes:
        if label == "background":
            continue
        if label not in seen:
            deduped.append(label)
            seen.add(label)

    missing = [label for label in discovered_labels if label not in seen]
    if missing:
        raise ValueError(f"Classes are missing labels found in JSON files: {missing}")

    return deduped


def collect_samples(
    image_index: dict[str, Path],
    json_index: dict[str, Path],
    include_unlabeled: bool,
) -> tuple[list[Sample], list[str], list[str]]:
    shared_stems = sorted(set(image_index) & set(json_index))
    image_only = sorted(set(image_index) - set(json_index))
    json_only = sorted(set(json_index) - set(image_index))

    samples: list[Sample] = []
    for stem in shared_stems:
        data = load_json(json_index[stem])
        labels = sorted(
            {
                str(shape.get("label", "")).strip()
                for shape in data.get("shapes", [])
                if str(shape.get("label", "")).strip()
            }
        )
        samples.append(
            Sample(
                stem=stem,
                image_path=image_index[stem],
                json_path=json_index[stem],
                labels=tuple(labels),
            )
        )

    if include_unlabeled:
        for stem in image_only:
            samples.append(
                Sample(
                    stem=stem,
                    image_path=image_index[stem],
                    json_path=None,
                    labels=tuple(),
                )
            )

    return samples, image_only, json_only


def prepare_output_dirs(out_dir: Path) -> dict[str, Path]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Please clear it first or choose another --out-dir."
        )

    dirs = {
        "images_train": out_dir / "images" / "train",
        "images_val": out_dir / "images" / "val",
        "labels_train": out_dir / "labels" / "train",
        "labels_val": out_dir / "labels" / "val",
        "meta": out_dir / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")
    if not samples:
        raise RuntimeError("No samples were collected for export.")
    if len(samples) == 1 or val_ratio == 0:
        return sorted(samples, key=lambda item: item.stem), []

    rng = random.Random(seed)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.split_key].append(sample)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for group_name in sorted(groups):
        group_samples = sorted(groups[group_name], key=lambda item: item.stem)
        rng.shuffle(group_samples)

        if len(group_samples) == 1:
            train_samples.extend(group_samples)
            continue

        val_count = int(round(len(group_samples) * val_ratio))
        val_count = max(1, val_count)
        val_count = min(len(group_samples) - 1, val_count)

        val_samples.extend(group_samples[:val_count])
        train_samples.extend(group_samples[val_count:])

    return sorted(train_samples, key=lambda item: item.stem), sorted(val_samples, key=lambda item: item.stem)


def normalize_points(points: list) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append((float(point[0]), float(point[1])))
    return normalized


def clamp_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        x = min(max(x, 0.0), float(max_x))
        y = min(max(y, 0.0), float(max_y))
        if not cleaned or abs(cleaned[-1][0] - x) > 1e-6 or abs(cleaned[-1][1] - y) > 1e-6:
            cleaned.append((x, y))

    if len(cleaned) > 1:
        first_x, first_y = cleaned[0]
        last_x, last_y = cleaned[-1]
        if abs(first_x - last_x) <= 1e-6 and abs(first_y - last_y) <= 1e-6:
            cleaned.pop()

    unique_points = {(round(x, 4), round(y, 4)) for x, y in cleaned}
    if len(unique_points) < 3:
        return []
    return cleaned


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def rectangle_to_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return []
    xs = [point[0] for point in points[:2]]
    ys = [point[1] for point in points[:2]]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def circle_to_polygon(points: list[tuple[float, float]], segments: int = 24) -> list[tuple[float, float]]:
    if len(points) < 2:
        return []
    (cx, cy), (px, py) = points[:2]
    radius = math.hypot(cx - px, cy - py)
    if radius <= 0:
        return []
    polygon: list[tuple[float, float]] = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        polygon.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return polygon


def shape_to_polygon(shape: dict, width: int, height: int) -> list[tuple[float, float]]:
    shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
    if shape_type not in SUPPORTED_SHAPES:
        raise ValueError(f"Unsupported shape_type for YOLO segmentation: {shape_type}")

    points = normalize_points(shape.get("points") or [])
    if shape_type == "polygon":
        polygon = points
    elif shape_type == "rectangle":
        polygon = rectangle_to_polygon(points)
    else:
        polygon = circle_to_polygon(points)

    polygon = clamp_points(polygon, width=width, height=height)
    if len(polygon) < 3 or polygon_area(polygon) < 1.0:
        return []
    return polygon


def polygon_to_yolo_line(class_id: int, polygon: list[tuple[float, float]], width: int, height: int) -> str:
    normalized_points: list[str] = []
    for x, y in polygon:
        normalized_points.append(f"{x / width:.6f}")
        normalized_points.append(f"{y / height:.6f}")
    return " ".join([str(class_id), *normalized_points])


def convert_sample(
    sample: Sample,
    label_to_id: dict[str, int],
) -> tuple[list[str], Counter]:
    if sample.json_path is None:
        return [], Counter()

    json_data = load_json(sample.json_path)
    with Image.open(sample.image_path) as image:
        width, height = image.size

    yolo_lines: list[str] = []
    class_counter: Counter = Counter()
    for shape in json_data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        if label not in label_to_id:
            raise ValueError(f"Found unknown label not declared in classes: {label}")

        polygon = shape_to_polygon(shape, width=width, height=height)
        if not polygon:
            continue

        class_id = label_to_id[label]
        yolo_lines.append(polygon_to_yolo_line(class_id, polygon, width=width, height=height))
        class_counter[label] += 1

    return yolo_lines, class_counter


def export_split(
    split_name: str,
    samples: list[Sample],
    image_out_dir: Path,
    label_out_dir: Path,
    label_to_id: dict[str, int],
    strict: bool,
) -> tuple[int, Counter, list[dict]]:
    success_count = 0
    object_counter: Counter = Counter()
    failures: list[dict] = []

    for index, sample in enumerate(samples, start=1):
        try:
            yolo_lines, class_counter = convert_sample(sample, label_to_id=label_to_id)
            shutil.copy2(sample.image_path, image_out_dir / sample.image_path.name)
            write_text_lines(label_out_dir / f"{sample.stem}.txt", yolo_lines)
            object_counter.update(class_counter)
            success_count += 1

            if index % 100 == 0 or index == len(samples):
                print(f"[{split_name}] processed {index}/{len(samples)}")
        except Exception as exc:
            failures.append(
                {
                    "split": split_name,
                    "stem": sample.stem,
                    "image": str(sample.image_path),
                    "json": str(sample.json_path) if sample.json_path else None,
                    "error": str(exc),
                }
            )
            print(f"[{split_name}] skipped {sample.stem}: {exc}")
            if strict:
                raise

    return success_count, object_counter, failures


def write_dataset_yaml(out_dir: Path, class_names: list[str]) -> Path:
    yaml_path = out_dir / "dataset.yaml"
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")

    with yaml_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return yaml_path


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    image_exts = tuple(ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip())

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_index = build_image_index(image_dir=image_dir, image_exts=image_exts)
    json_index = build_json_index(label_dir=label_dir)

    discovered_labels = discover_labels([json_index[stem] for stem in sorted(set(image_index) & set(json_index))])
    if not discovered_labels:
        raise RuntimeError("No labels were discovered from the matched LabelMe JSON files.")

    class_names = parse_classes(args.classes, discovered_labels=discovered_labels)
    label_to_id = {name: index for index, name in enumerate(class_names)}

    samples, image_only, json_only = collect_samples(
        image_index=image_index,
        json_index=json_index,
        include_unlabeled=args.include_unlabeled,
    )
    train_samples, val_samples = split_samples(samples=samples, val_ratio=args.val_ratio, seed=args.seed)
    dirs = prepare_output_dirs(out_dir=out_dir)

    print("Starting LabelMe -> YOLO segmentation conversion")
    print(f"Collected samples: {len(samples)}")
    print(f"Matched image/json pairs: {len(samples) - (len(image_only) if args.include_unlabeled else 0)}")
    print(f"Images without JSON: {len(image_only)}")
    print(f"JSON without image: {len(json_only)}")
    print(f"Classes: {label_to_id}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_success, train_objects, train_failures = export_split(
        split_name="train",
        samples=train_samples,
        image_out_dir=dirs["images_train"],
        label_out_dir=dirs["labels_train"],
        label_to_id=label_to_id,
        strict=args.strict,
    )
    val_success, val_objects, val_failures = export_split(
        split_name="val",
        samples=val_samples,
        image_out_dir=dirs["images_val"],
        label_out_dir=dirs["labels_val"],
        label_to_id=label_to_id,
        strict=args.strict,
    )

    failures = train_failures + val_failures
    dataset_yaml = write_dataset_yaml(out_dir=out_dir, class_names=class_names)

    summary = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "out_dir": str(out_dir),
        "dataset_yaml": str(dataset_yaml),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "include_unlabeled": args.include_unlabeled,
        "classes": class_names,
        "label_to_id": label_to_id,
        "total_images": len(image_index),
        "total_jsons": len(json_index),
        "matched_pairs": len(set(image_index) & set(json_index)),
        "exported_samples": train_success + val_success,
        "train_samples": train_success,
        "val_samples": val_success,
        "image_only_count": len(image_only),
        "json_only_count": len(json_only),
        "failure_count": len(failures),
        "train_objects": dict(train_objects),
        "val_objects": dict(val_objects),
        "split_distribution": dict(Counter(sample.split_key for sample in samples)),
    }

    dump_json(dirs["meta"] / "dataset_summary.json", summary)
    dump_json(dirs["meta"] / "failures.json", {"items": failures})
    dump_json(dirs["meta"] / "label_to_id.json", label_to_id)
    write_text_lines(dirs["meta"] / "classes.txt", class_names)
    write_text_lines(dirs["meta"] / "train.txt", [sample.stem for sample in train_samples])
    write_text_lines(dirs["meta"] / "val.txt", [sample.stem for sample in val_samples])
    write_text_lines(dirs["meta"] / "image_only.txt", image_only)
    write_text_lines(dirs["meta"] / "json_only.txt", json_only)

    print("Conversion finished")
    print(f"Dataset root: {out_dir}")
    print(f"Dataset yaml: {dataset_yaml}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
