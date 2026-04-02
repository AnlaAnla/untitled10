import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_IMAGE_DIR = r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project\imgs"
DEFAULT_LABEL_DIR = r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project\labels"
DEFAULT_OUT_DIR = r"C:\Code\ML\Image\_SEGMENT\video_card_hand_project_dataset"
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SUPPORTED_SHAPE_TYPES = {"polygon", "rectangle", "circle", "line", "linestrip", "point"}


@dataclass(frozen=True)
class SamplePair:
    stem: str
    image_path: Path
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 LabelMe 标注转换成 SegFormer 训练脚本可用的数据集目录。"
    )
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="原始图片目录")
    parser.add_argument("--label-dir", type=str, default=DEFAULT_LABEL_DIR, help="LabelMe JSON 目录")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="输出数据集目录")
    parser.add_argument(
        "--classes",
        type=str,
        default="auto",
        help='类别定义，默认 "auto" 自动从 JSON 提取；也可手动传 "background,card,hand"',
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，范围 [0, 1)")
    parser.add_argument("--seed", type=int, default=42, help="划分 train/val 的随机种子")
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help="允许的图片后缀，逗号分隔",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到单个样本异常时立即报错退出；默认跳过异常样本并继续处理",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_classes(class_text: str, discovered_labels: list[str]) -> list[str]:
    if class_text.strip().lower() == "auto":
        classes = ["background", *discovered_labels]
    else:
        classes = [name.strip() for name in class_text.split(",") if name.strip()]
        if not classes:
            raise ValueError("--classes 不能为空。")
        if "background" not in classes:
            classes = ["background", *classes]
        elif classes[0] != "background":
            classes = ["background", *[name for name in classes if name != "background"]]

        missing = [label for label in discovered_labels if label not in classes]
        if missing:
            raise ValueError(
                f"--classes 未覆盖 JSON 中的全部标签，缺失: {missing}。"
            )

    deduped = []
    seen = set()
    for name in classes:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def build_image_index(image_dir: Path, image_exts: tuple[str, ...]) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in image_exts:
            continue
        if path.stem in image_index:
            raise ValueError(
                f"图片目录中存在同名不同后缀文件，无法唯一匹配: {path.stem}"
            )
        image_index[path.stem] = path
    return image_index


def build_json_index(label_dir: Path) -> dict[str, Path]:
    json_index: dict[str, Path] = {}
    for path in sorted(label_dir.rglob("*.json")):
        if path.stem in json_index:
            raise ValueError(
                f"标注目录中存在同名 JSON，无法唯一匹配: {path.stem}"
            )
        json_index[path.stem] = path
    return json_index


def collect_pairs(image_index: dict[str, Path], json_index: dict[str, Path]) -> tuple[list[SamplePair], list[str], list[str]]:
    shared_stems = sorted(set(image_index) & set(json_index))
    image_only = sorted(set(image_index) - set(json_index))
    json_only = sorted(set(json_index) - set(image_index))
    pairs = [
        SamplePair(stem=stem, image_path=image_index[stem], json_path=json_index[stem])
        for stem in shared_stems
    ]
    return pairs, image_only, json_only


def discover_labels(pairs: list[SamplePair]) -> list[str]:
    labels: list[str] = []
    seen = set()
    for pair in pairs:
        data = load_json(pair.json_path)
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def validate_args(args: argparse.Namespace, pairs: list[SamplePair]) -> None:
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围内。")
    if not pairs:
        raise RuntimeError("未找到任何同名图片/JSON 配对样本。")


def prepare_output_dirs(out_dir: Path) -> dict[str, Path]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且非空: {out_dir}\n"
            "为避免与旧数据混在一起，请先清空该目录或更换 --out-dir。"
        )

    dirs = {
        "train_image": out_dir / "images" / "train",
        "val_image": out_dir / "images" / "val",
        "train_mask": out_dir / "masks" / "train",
        "val_mask": out_dir / "masks" / "val",
        "meta": out_dir / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def split_pairs(pairs: list[SamplePair], val_ratio: float, seed: int) -> tuple[list[SamplePair], list[SamplePair]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    if val_ratio <= 0 or len(shuffled) == 1:
        return sorted(shuffled, key=lambda item: item.stem), []

    val_count = int(round(len(shuffled) * val_ratio))
    val_count = max(1, val_count)
    val_count = min(len(shuffled) - 1, val_count)

    val_stems = {pair.stem for pair in shuffled[:val_count]}
    train_pairs = sorted((pair for pair in pairs if pair.stem not in val_stems), key=lambda item: item.stem)
    val_pairs = sorted((pair for pair in pairs if pair.stem in val_stems), key=lambda item: item.stem)
    return train_pairs, val_pairs


def normalize_points(points: list) -> list[tuple[float, float]]:
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append((float(point[0]), float(point[1])))
    return normalized


def draw_shape(mask_draw: ImageDraw.ImageDraw, shape: dict, class_id: int) -> None:
    shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
    points = normalize_points(shape.get("points") or [])

    if shape_type not in SUPPORTED_SHAPE_TYPES:
        raise ValueError(f"暂不支持的 shape_type: {shape_type}")

    if shape_type == "polygon":
        if len(points) >= 3:
            mask_draw.polygon(points, fill=class_id, outline=class_id)
        elif len(points) == 2:
            mask_draw.line(points, fill=class_id, width=1)
        return

    if shape_type == "rectangle":
        if len(points) >= 2:
            (x1, y1), (x2, y2) = points[:2]
            mask_draw.rectangle((x1, y1, x2, y2), fill=class_id, outline=class_id)
        return

    if shape_type == "circle":
        if len(points) >= 2:
            (cx, cy), (px, py) = points[:2]
            radius = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            mask_draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=class_id,
                outline=class_id,
            )
        return

    if shape_type in {"line", "linestrip"}:
        if len(points) >= 2:
            mask_draw.line(points, fill=class_id, width=5)
        return

    if shape_type == "point" and points:
        x, y = points[0]
        radius = 3
        mask_draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=class_id,
            outline=class_id,
        )


def build_mask(json_data: dict, image_size: tuple[int, int], label2id: dict[str, int]) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)

    for shape in json_data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        if label not in label2id:
            raise ValueError(f"发现未知标签: {label}")
        draw_shape(draw, shape, label2id[label])

    return mask


def copy_image(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)


def write_text_lines(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")


def dump_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def convert_split(
    split_name: str,
    pairs: list[SamplePair],
    image_out_dir: Path,
    mask_out_dir: Path,
    label2id: dict[str, int],
    strict: bool,
) -> tuple[int, list[dict]]:
    success_count = 0
    failures: list[dict] = []

    for idx, pair in enumerate(pairs, start=1):
        try:
            json_data = load_json(pair.json_path)
            with Image.open(pair.image_path) as img:
                image_size = img.size

            mask = build_mask(json_data, image_size, label2id)

            image_dst = image_out_dir / pair.image_path.name
            mask_dst = mask_out_dir / f"{pair.stem}.png"
            copy_image(pair.image_path, image_dst)
            mask.save(mask_dst)
            success_count += 1

            if idx % 100 == 0 or idx == len(pairs):
                print(f"[{split_name}] 已处理 {idx}/{len(pairs)}")
        except Exception as exc:
            failures.append(
                {
                    "split": split_name,
                    "stem": pair.stem,
                    "image": str(pair.image_path),
                    "json": str(pair.json_path),
                    "error": str(exc),
                }
            )
            print(f"[{split_name}] 跳过异常样本 {pair.stem}: {exc}")
            if strict:
                raise

    return success_count, failures


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    image_exts = tuple(ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip())

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {label_dir}")

    image_index = build_image_index(image_dir, image_exts)
    json_index = build_json_index(label_dir)
    pairs, image_only, json_only = collect_pairs(image_index, json_index)
    validate_args(args, pairs)

    discovered_labels = discover_labels(pairs)
    if not discovered_labels:
        raise RuntimeError("所有 JSON 都没有有效 shapes，无法生成训练数据。")

    classes = parse_classes(args.classes, discovered_labels)
    label2id = {name: idx for idx, name in enumerate(classes)}
    dirs = prepare_output_dirs(out_dir)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    print("开始转换 LabelMe 数据集")
    print(f"配对样本数: {len(pairs)}")
    print(f"仅有图片无 JSON: {len(image_only)}")
    print(f"仅有 JSON 无图片: {len(json_only)}")
    print(f"类别映射: {label2id}")
    print(f"训练集: {len(train_pairs)}，验证集: {len(val_pairs)}")

    train_success, train_failures = convert_split(
        split_name="train",
        pairs=train_pairs,
        image_out_dir=dirs["train_image"],
        mask_out_dir=dirs["train_mask"],
        label2id=label2id,
        strict=args.strict,
    )
    val_success, val_failures = convert_split(
        split_name="val",
        pairs=val_pairs,
        image_out_dir=dirs["val_image"],
        mask_out_dir=dirs["val_mask"],
        label2id=label2id,
        strict=args.strict,
    )

    failures = train_failures + val_failures
    summary = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "out_dir": str(out_dir),
        "classes": classes,
        "label2id": label2id,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "matched_pairs": len(pairs),
        "train_count": train_success,
        "val_count": val_success,
        "image_only_count": len(image_only),
        "json_only_count": len(json_only),
        "failure_count": len(failures),
    }

    dump_json(dirs["meta"] / "dataset_summary.json", summary)
    dump_json(dirs["meta"] / "label2id.json", label2id)
    dump_json(dirs["meta"] / "failures.json", {"items": failures})
    write_text_lines(dirs["meta"] / "classes.txt", classes)
    write_text_lines(dirs["meta"] / "train.txt", [pair.stem for pair in train_pairs])
    write_text_lines(dirs["meta"] / "val.txt", [pair.stem for pair in val_pairs])
    write_text_lines(dirs["meta"] / "image_only.txt", image_only)
    write_text_lines(dirs["meta"] / "json_only.txt", json_only)

    print("转换完成")
    print(f"输出目录: {out_dir}")
    print(f"train 成功: {train_success}")
    print(f"val 成功: {val_success}")
    print(f"失败样本: {len(failures)}")
    print("训练时可使用如下类别参数：")
    print(f"--classes {','.join(classes)}")


if __name__ == "__main__":
    main()
