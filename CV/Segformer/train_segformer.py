import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


LOGGER_NAME = "segformer_train"


# -----------------------------
# 日志与工具函数
# -----------------------------


def setup_logger(out_dir: Path) -> logging.Logger:
    # 同时输出到控制台和日志文件，方便长时间训练后回看。
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_json(logger: logging.Logger, title: str, payload: dict) -> None:
    logger.info("%s\n%s", title, json.dumps(payload, ensure_ascii=False, indent=2))


def set_seed(seed: int) -> None:
    # 固定主进程和 CUDA 的随机种子，尽量提升复现性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # DataLoader 子进程也要同步随机种子，否则增强结果不可控。
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device_name: str, logger: logging.Logger) -> torch.device:
    # 优先尊重用户指定设备，不可用时自动回退到 CPU。
    try:
        device = torch.device(device_name)
    except Exception as exc:
        raise ValueError(f"无法识别设备参数: {device_name}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("未检测到 CUDA，自动切换到 CPU。")
        return torch.device("cpu")

    if device.type == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            logger.warning("当前环境不支持 MPS，自动切换到 CPU。")
            return torch.device("cpu")

    return device


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scaler,
    class_names: list,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    # 统一 checkpoint 格式，便于断点续训和结果追溯。
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "classes": class_names,
            "best_miou": best_miou,
            "args": vars(args),
        },
        path,
    )


@dataclass
class InstanceStats:
    # 实例级统计量，用于近似计算 Precision / Recall / F1。
    tp: int = 0
    fp: int = 0
    fn: int = 0


# -----------------------------
# 数据集与增强
# -----------------------------


class SegTransform:
    # 训练/验证阶段的尺寸处理与基础增强。
    def __init__(
        self,
        img_size: int,
        train: bool,
        scale_min: float,
        scale_max: float,
        hflip: float,
        vflip: float,
        color_jitter: tuple,
        pad_value: int = 0,
        mask_pad_value: int = 0,
    ) -> None:
        self.img_size = img_size
        self.train = train
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.pad_value = pad_value
        self.mask_pad_value = mask_pad_value

    def _random_scale(self, image: Image.Image, mask: Image.Image) -> tuple:
        # 图像和掩码必须同步缩放，否则像素对不齐。
        if self.scale_min == 1.0 and self.scale_max == 1.0:
            return image, mask
        scale = random.uniform(self.scale_min, self.scale_max)
        new_w = max(1, int(image.width * scale))
        new_h = max(1, int(image.height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask

    def _maybe_flip(self, image: Image.Image, mask: Image.Image) -> tuple:
        # 翻转时图像和掩码也要一起翻。
        if self.hflip > 0 and random.random() < self.hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.vflip > 0 and random.random() < self.vflip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def _random_crop_or_pad(self, image: Image.Image, mask: Image.Image) -> tuple:
        # 先补边到最小尺寸，再随机裁剪到目标尺寸。
        target_w = self.img_size
        target_h = self.img_size

        pad_w = max(0, target_w - image.width)
        pad_h = max(0, target_h - image.height)
        if pad_w > 0 or pad_h > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            image = ImageOps.expand(
                image,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.pad_value,
            )
            mask = ImageOps.expand(
                mask,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.mask_pad_value,
            )

        if image.width == target_w and image.height == target_h:
            return image, mask

        left = random.randint(0, image.width - target_w)
        top = random.randint(0, image.height - target_h)
        image = image.crop((left, top, left + target_w, top + target_h))
        mask = mask.crop((left, top, left + target_w, top + target_h))
        return image, mask

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        # 对直播间常见的光照和色温波动做轻量扰动。
        b, c, s = self.color_jitter
        if b > 0:
            factor = random.uniform(max(0, 1 - b), 1 + b)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if c > 0:
            factor = random.uniform(max(0, 1 - c), 1 + c)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if s > 0:
            factor = random.uniform(max(0, 1 - s), 1 + s)
            image = ImageEnhance.Color(image).enhance(factor)
        return image

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        if self.train:
            # 训练阶段启用随机增强，增强对手势、角度和光照变化的鲁棒性。
            image, mask = self._random_scale(image, mask)
            image, mask = self._maybe_flip(image, mask)
            image, mask = self._random_crop_or_pad(image, mask)
            image = self._color_jitter(image)
        else:
            # 验证阶段保持确定性，方便横向比较指标。
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        return image, mask


class SegDataset(Dataset):
    # 语义分割数据集读取器，默认目录结构为 images/{split} 和 masks/{split}。
    def __init__(
        self,
        root: str,
        split: str,
        img_exts: tuple,
        mask_exts: tuple,
        transforms: SegTransform,
        mean: tuple,
        std: tuple,
        ignore_index: int,
        num_classes: int,
        limit: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        self.img_exts = img_exts
        self.mask_exts = mask_exts
        self.transforms = transforms
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.items = self._scan()
        if limit > 0:
            self.items = self.items[:limit]

        if len(self.items) == 0:
            raise RuntimeError(f"未在 {self.img_dir} 和 {self.mask_dir} 中找到可配对的数据。")

    def _scan(self) -> list:
        # 根据文件名 stem 匹配 image 与 mask，例如 0001.jpg -> 0001.png。
        items = []
        for ext in self.img_exts:
            for img_path in self.img_dir.glob(f"*{ext}"):
                stem = img_path.stem
                mask_path = None
                for mext in self.mask_exts:
                    candidate = self.mask_dir / f"{stem}{mext}"
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path is not None:
                    items.append((img_path, mask_path))
        items.sort()
        return items

    @staticmethod
    def _load_mask(mask_path: Path) -> Image.Image:
        # P 模式（调色板 PNG）必须保留原始类别索引，不能直接转成灰度。
        with Image.open(mask_path) as mask_file:
            if mask_file.mode in {"L", "P"}:
                return mask_file.copy()

            mask_array = np.array(mask_file)

        if mask_array.ndim != 2:
            raise ValueError(
                f"掩码文件 {mask_path} 不是单通道类别索引图，当前形状为 {mask_array.shape}。"
                "请先把标注图转换为像素值等于类别 id 的单通道图。"
            )

        return Image.fromarray(mask_array.astype(np.uint8), mode="L")

    def _validate_mask_values(self, mask_array: np.ndarray, mask_path: Path) -> None:
        # 提前发现越界类别值，避免训练半天后才知道标注有问题。
        invalid = (mask_array != self.ignore_index) & (
            (mask_array < 0) | (mask_array >= self.num_classes)
        )
        if invalid.any():
            invalid_values = np.unique(mask_array[invalid]).tolist()[:10]
            raise ValueError(
                f"掩码文件 {mask_path} 中存在越界类别值 {invalid_values}。"
                f"当前类别数为 {self.num_classes}，合法范围应为 [0, {self.num_classes - 1}]，"
                f"忽略标签为 {self.ignore_index}。"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        # 读取图像与掩码，并转换为模型可直接消费的张量格式。
        img_path, mask_path = self.items[idx]

        with Image.open(img_path) as image_file:
            image = image_file.convert("RGB")
        mask = self._load_mask(mask_path)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask_array = np.array(mask, dtype=np.int64)
        self._validate_mask_values(mask_array, mask_path)
        mask = torch.from_numpy(mask_array)

        return image, mask


# -----------------------------
# 损失函数
# -----------------------------


def dice_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int, include_background: bool) -> torch.Tensor:
    # Soft Dice Loss 对类不平衡更友好，适合卡牌区域大、手部区域小的情况。
    probs = F.softmax(logits, dim=1)

    if ignore_index is not None:
        valid = labels != ignore_index
        labels = labels.clone()
        labels[~valid] = 0
    else:
        valid = torch.ones_like(labels, dtype=torch.bool)

    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    if not include_background and num_classes > 1:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    intersection = (probs * one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int, gamma: float, alpha: float) -> torch.Tensor:
    # Focal Loss 用来突出难分样本，例如手指边缘、反光卡套等区域。
    ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce)
    loss = (alpha * (1 - pt) ** gamma * ce)
    return loss.mean()


def tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    alpha: float,
    beta: float,
    include_background: bool,
) -> torch.Tensor:
    # Tversky Loss 是 Dice 的泛化形式，可分别控制 FP / FN 的惩罚力度。
    probs = F.softmax(logits, dim=1)

    if ignore_index is not None:
        valid = labels != ignore_index
        labels = labels.clone()
        labels[~valid] = 0
    else:
        valid = torch.ones_like(labels, dtype=torch.bool)

    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    if not include_background and num_classes > 1:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    tp = (probs * one_hot).sum(dim=(2, 3))
    fp = (probs * (1 - one_hot)).sum(dim=(2, 3))
    fn = ((1 - probs) * one_hot).sum(dim=(2, 3))

    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return 1 - tversky.mean()


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    loss_names: list,
    loss_weights: list,
    class_weights: torch.Tensor,
    include_background: bool,
    focal_gamma: float,
    focal_alpha: float,
    tversky_alpha: float,
    tversky_beta: float,
) -> tuple:
    # 支持多种损失函数加权求和，便于后续针对小目标继续调参。
    total = 0.0
    details = {}

    for name, weight in zip(loss_names, loss_weights):
        if name == "ce":
            loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
        elif name == "dice":
            loss = dice_loss(logits, labels, num_classes, ignore_index, include_background)
        elif name == "focal":
            loss = focal_loss(logits, labels, ignore_index, focal_gamma, focal_alpha)
        elif name == "tversky":
            loss = tversky_loss(logits, labels, num_classes, ignore_index, tversky_alpha, tversky_beta, include_background)
        else:
            raise ValueError(f"不支持的损失函数: {name}")

        total = total + weight * loss
        details[name] = float(loss.detach().item())

    return total, details


# -----------------------------
# 语义分割指标
# -----------------------------


def update_confusion(conf, preds, labels, num_classes, ignore_index):
    # 使用混淆矩阵统一累计整套语义分割指标。
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    hist = torch.bincount(
        num_classes * labels + preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    conf += hist.cpu().numpy()


def compute_metrics(conf):
    # 若某个类别在 GT 和预测中都不存在，则从 mIoU / mDice 均值中排除。
    conf = conf.astype(np.float64)
    diag = np.diag(conf)
    total = conf.sum()
    union = conf.sum(1) + conf.sum(0) - diag
    support = conf.sum(1) + conf.sum(0)
    acc = float(diag.sum() / max(total, 1.0))
    iou = np.divide(
        diag,
        union,
        out=np.full(diag.shape, np.nan, dtype=np.float64),
        where=union > 0,
    )
    dice = np.divide(
        2 * diag,
        support,
        out=np.full(diag.shape, np.nan, dtype=np.float64),
        where=support > 0,
    )
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    mdice = float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0
    return acc, miou, mdice, iou, dice


# -----------------------------
# 实例级指标(连通域近似)
# -----------------------------


try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def connected_components(binary_mask: np.ndarray) -> tuple:
    # 返回 (labels, count)，其中 labels=0 表示背景。
    if _HAS_CV2:
        num, labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
        return labels, num - 1
    if _HAS_SCIPY:
        labels, num = ndi.label(binary_mask)
        return labels, num

    # 纯 numpy 兜底，速度更慢，但不依赖额外库。
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] and labels[y, x] == 0:
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary_mask[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = current
                                    stack.append((ny, nx))
    return labels, current


def instance_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, iou_thr: float, min_area: int) -> InstanceStats:
    # 近似实例统计：把每个连通域视为一个实例。
    stats = InstanceStats()

    for cls in range(1, num_classes):
        pred_labels, pred_n = connected_components(pred == cls)
        gt_labels, gt_n = connected_components(gt == cls)

        # 过滤过小的连通域，减少碎片噪声对实例指标的影响。
        pred_areas = [0] * (pred_n + 1)
        gt_areas = [0] * (gt_n + 1)
        for pid in range(1, pred_n + 1):
            pred_areas[pid] = int((pred_labels == pid).sum())
        for gid in range(1, gt_n + 1):
            gt_areas[gid] = int((gt_labels == gid).sum())

        pred_valid = [pid for pid in range(1, pred_n + 1) if pred_areas[pid] >= min_area]
        gt_valid = [gid for gid in range(1, gt_n + 1) if gt_areas[gid] >= min_area]

        matched_pred = set()
        for gid in gt_valid:
            gt_mask = gt_labels == gid
            best_iou = 0.0
            best_pid = None
            for pid in pred_valid:
                if pid in matched_pred:
                    continue
                pred_mask = pred_labels == pid
                inter = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid
            if best_iou >= iou_thr and best_pid is not None:
                stats.tp += 1
                matched_pred.add(best_pid)
            else:
                stats.fn += 1

        stats.fp += max(0, len(pred_valid) - len(matched_pred))

    return stats


# -----------------------------
# 训练与验证
# -----------------------------


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    num_classes,
    ignore_index,
    loss_names,
    loss_weights,
    class_weights,
    include_background,
    focal_gamma,
    focal_alpha,
    tversky_alpha,
    tversky_beta,
    grad_accum,
    print_freq,
    use_amp,
    logger,
    epoch,
    total_epochs,
):
    # 单轮训练逻辑，包含梯度累积和中文日志。
    model.train()
    total_loss = 0.0
    step_count = 0
    num_steps = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(pixel_values=images)
            logits = outputs.logits
            # 输出尺寸与标签对齐
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss, details = compute_loss(
                logits,
                labels,
                num_classes,
                ignore_index,
                loss_names,
                loss_weights,
                class_weights,
                include_background,
                focal_gamma,
                focal_alpha,
                tversky_alpha,
                tversky_beta,
            )
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        # 修复最后一组样本不足 grad_accum 时不会执行优化器更新的问题。
        should_step = (step % grad_accum == 0) or (step == num_steps)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum
        step_count += 1

        if print_freq > 0 and (step % print_freq == 0 or step == num_steps):
            avg_loss = total_loss / max(step_count, 1)
            lr = optimizer.param_groups[0]["lr"]
            detail_text = "".join(f" {k}:{v:.4f}" for k, v in details.items())
            logger.info(
                "训练中 第 %d/%d 轮 第 %d/%d 步 平均损失 %.4f 学习率 %.6g%s",
                epoch,
                total_epochs,
                step,
                num_steps,
                avg_loss,
                lr,
                detail_text,
            )

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    ignore_index,
    instance_metric,
    inst_iou_thr,
    inst_min_area,
):
    # 验证阶段输出语义分割指标，并可选计算近似实例指标。
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    inst_stats = InstanceStats()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        update_confusion(conf, preds, labels, num_classes, ignore_index)

        if instance_metric:
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, g in zip(preds_np, labels_np):
                stats = instance_metrics(p, g, num_classes, inst_iou_thr, inst_min_area)
                inst_stats.tp += stats.tp
                inst_stats.fp += stats.fp
                inst_stats.fn += stats.fn

    acc, miou, mdice, iou, dice = compute_metrics(conf)

    inst_result = None
    if instance_metric:
        tp, fp, fn = inst_stats.tp, inst_stats.fp, inst_stats.fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        inst_acc = tp / max(tp + fp + fn, 1)
        inst_result = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": inst_acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return acc, miou, mdice, iou, dice, inst_result


# -----------------------------
# 参数与入口
# -----------------------------


def parse_args():
    # 先只解析 --cfg，避免配置文件里的 data 被 required 提前拦住。
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cfg", type=str, default="", help="可选的 JSON/YAML 配置文件")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="SegFormer 语义分割训练脚本（支持实例风格指标）")
    parser.add_argument("--cfg", type=str, default="", help="可选的 JSON/YAML 配置文件，使用扁平键值")
    parser.add_argument("--data", type=str, default="", help="数据集根目录，目录内需包含 images/ 与 masks/")
    parser.add_argument("--classes", type=str, default="background,card,hand", help="类别名，逗号分隔，例如 background,card,hand")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Hugging Face 预训练模型名称或本地路径")

    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="训练 batch 大小")
    parser.add_argument("--img-size", type=int, default=512, help="输入图像尺寸")
    parser.add_argument("--lr", type=float, default=6e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")

    parser.add_argument("--device", type=str, default="cuda", help="训练设备，例如 cuda、cuda:0、cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性训练，牺牲少量速度换稳定")

    parser.add_argument("--img-exts", type=str, default=".jpg,.jpeg,.png", help="图像后缀，逗号分隔")
    parser.add_argument("--mask-exts", type=str, default=".png,.bmp,.tif,.tiff", help="掩码后缀，逗号分隔")
    parser.add_argument("--ignore-index", type=int, default=255, help="忽略标签值")

    parser.add_argument("--scale", type=float, nargs=2, default=(0.8, 1.2), help="随机缩放范围")
    parser.add_argument("--hflip", type=float, default=0.5, help="水平翻转概率")
    parser.add_argument("--vflip", type=float, default=0.0, help="垂直翻转概率")
    parser.add_argument("--color-jitter", type=float, nargs=3, default=(0.2, 0.2, 0.2), help="亮度/对比度/饱和度扰动")

    parser.add_argument("--loss", type=str, nargs="+", default=["ce", "dice"], help="损失函数，可选 ce dice focal tversky")
    parser.add_argument("--loss-weights", type=float, nargs="+", default=[1.0, 1.0], help="与 --loss 对应的权重")
    parser.add_argument("--class-weights", type=float, nargs="*", default=None, help="交叉熵的类别权重")
    parser.add_argument("--dice-include-bg", action="store_true", help="Dice/Tversky 是否包含背景类")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss 的 gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal Loss 的 alpha")
    parser.add_argument("--tversky-alpha", type=float, default=0.5, help="Tversky Loss 的 alpha")
    parser.add_argument("--tversky-beta", type=float, default=0.5, help="Tversky Loss 的 beta")

    parser.add_argument("--instance-metric", action="store_true", help="验证时计算近似实例级指标")
    parser.add_argument("--inst-iou-thr", type=float, default=0.5, help="实例匹配 IoU 阈值")
    parser.add_argument("--inst-min-area", type=int, default=32, help="实例最小连通域面积")

    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 混合精度训练（仅 CUDA 可用）")
    parser.add_argument("--print-freq", type=int, default=20, help="每多少步打印一次训练日志，0 表示关闭")
    parser.add_argument("--eval-every", type=int, default=1, help="每多少轮做一次验证，0 表示关闭")
    parser.add_argument("--save-every", type=int, default=1, help="每多少轮保存一次 checkpoint，0 表示关闭")
    parser.add_argument("--out-dir", type=str, default="runs/segformer", help="输出目录")
    parser.add_argument("--resume", type=str, default="", help="断点续训 checkpoint 路径")
    parser.add_argument("--limit-train", type=int, default=0, help="限制训练集样本数，0 表示不限制")
    parser.add_argument("--limit-val", type=int, default=0, help="限制验证集样本数，0 表示不限制")

    if pre_args.cfg:
        cfg = load_cfg(pre_args.cfg)
        if not isinstance(cfg, dict):
            raise ValueError("配置文件必须是扁平的键值对字典。")
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if not args.data:
        parser.error("请通过 --data 或配置文件提供数据集根目录。")

    return args


def load_cfg(path: str) -> dict:
    # 支持 JSON / YAML 两种配置文件格式。
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in [".json"]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    if suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("读取 YAML 配置需要先安装 PyYAML。") from exc
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    raise ValueError("配置文件必须是 .json 或 .yaml/.yml 格式。")


def validate_args(args: argparse.Namespace, num_classes: int) -> None:
    # 统一检查参数边界，避免训练开始后才报错。
    if num_classes < 2:
        raise ValueError("至少需要 2 个类别，且必须包含 background。")

    if len(args.loss) != len(args.loss_weights):
        raise ValueError("--loss 与 --loss-weights 的长度必须一致。")

    if args.class_weights and len(args.class_weights) != num_classes:
        raise ValueError("--class-weights 的长度必须和类别数一致。")

    if args.grad_accum < 1:
        raise ValueError("--grad-accum 必须大于等于 1。")

    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于等于 1。")

    if args.img_size < 32:
        raise ValueError("--img-size 太小，建议至少为 32。")

    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数。")

    if args.eval_every < 0 or args.save_every < 0:
        raise ValueError("--eval-every 和 --save-every 不能为负数。")

    if len(args.scale) != 2 or args.scale[0] <= 0 or args.scale[1] < args.scale[0]:
        raise ValueError("--scale 必须是两个正数，且满足 min <= max。")

    if len(args.color_jitter) != 3:
        raise ValueError("--color-jitter 需要传入 3 个值：亮度、对比度、饱和度。")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    num_classes = len(class_names)
    validate_args(args, num_classes)

    device = resolve_device(args.device, logger)
    use_amp = bool(args.fp16 and device.type == "cuda")

    if args.fp16 and not use_amp:
        logger.warning("当前设备不是 CUDA，已自动关闭 FP16 混合精度。")

    if args.deterministic:
        # 牺牲少量性能换更稳定的实验复现。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == "cuda":
        # 固定尺寸训练时通常可以得到更快的卷积实现。
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    logger.info("开始训练 SegFormer 分割模型。")
    logger.info("运行设备: %s", device)
    logger.info("输出目录: %s", out_dir.resolve())
    if args.instance_metric and not (_HAS_CV2 or _HAS_SCIPY):
        logger.warning("未安装 OpenCV / SciPy，实例指标将退回纯 numpy 连通域实现，验证速度会较慢。")

    # 使用预训练模型自带的 mean / std，保证输入分布一致。
    processor = AutoImageProcessor.from_pretrained(args.model)
    mean = tuple(float(x) for x in processor.image_mean)
    std = tuple(float(x) for x in processor.image_std)

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    # 加载 SegFormer，并替换成当前任务的分类头。
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 数据集目录结构示例：
    # data_root/
    #   images/train/*.jpg
    #   images/val/*.jpg
    #   masks/train/*.png
    #   masks/val/*.png
    train_tf = SegTransform(
        img_size=args.img_size,
        train=True,
        scale_min=args.scale[0],
        scale_max=args.scale[1],
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=tuple(args.color_jitter),
    )
    val_tf = SegTransform(
        img_size=args.img_size,
        train=False,
        scale_min=1.0,
        scale_max=1.0,
        hflip=0.0,
        vflip=0.0,
        color_jitter=(0.0, 0.0, 0.0),
    )

    train_ds = SegDataset(
        root=args.data,
        split="train",
        img_exts=tuple(x.strip() for x in args.img_exts.split(",") if x.strip()),
        mask_exts=tuple(x.strip() for x in args.mask_exts.split(",") if x.strip()),
        transforms=train_tf,
        mean=mean,
        std=std,
        ignore_index=args.ignore_index,
        num_classes=num_classes,
        limit=args.limit_train,
    )
    val_ds = SegDataset(
        root=args.data,
        split="val",
        img_exts=tuple(x.strip() for x in args.img_exts.split(",") if x.strip()),
        mask_exts=tuple(x.strip() for x in args.mask_exts.split(",") if x.strip()),
        transforms=val_tf,
        mean=mean,
        std=std,
        ignore_index=args.ignore_index,
        num_classes=num_classes,
        limit=args.limit_val,
    )

    logger.info(
        "数据集加载完成：训练集 %d 张，验证集 %d 张，类别=%s",
        len(train_ds),
        len(val_ds),
        " / ".join(class_names),
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent_workers,
    )

    logger.info(
        "DataLoader 设置：train_batch=%d, val_batch=%d, num_workers=%d, pin_memory=%s, persistent_workers=%s",
        args.batch_size,
        max(1, args.batch_size // 2),
        args.num_workers,
        pin_memory,
        persistent_workers,
    )

    # 优化器与混合精度。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
        logger.info("已启用类别权重：%s", class_weights.detach().cpu().tolist())

    # 断点续训。
    start_epoch = 1
    best_miou = -1.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"断点文件不存在: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            logger.warning("断点中未包含 optimizer 状态，将使用当前优化器配置继续训练。")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("best_miou", -1.0))
        logger.info(
            "已从断点恢复：%s，从第 %d 轮继续训练，当前最佳 mIoU=%.4f",
            resume_path,
            start_epoch,
            max(best_miou, 0.0),
        )

    log_json(
        logger,
        "训练配置摘要：",
        {
            "data": args.data,
            "classes": class_names,
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "loss_weights": args.loss_weights,
            "scale": list(args.scale),
            "color_jitter": list(args.color_jitter),
            "grad_accum": args.grad_accum,
            "fp16_enabled": use_amp,
            "instance_metric": args.instance_metric,
            "ignore_index": args.ignore_index,
        },
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            num_classes,
            args.ignore_index,
            args.loss,
            args.loss_weights,
            class_weights,
            args.dice_include_bg,
            args.focal_gamma,
            args.focal_alpha,
            args.tversky_alpha,
            args.tversky_beta,
            args.grad_accum,
            args.print_freq,
            use_amp,
            logger,
            epoch,
            args.epochs,
        )

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            acc, miou, mdice, iou, dice, inst = evaluate(
                model,
                val_loader,
                device,
                num_classes,
                args.ignore_index,
                args.instance_metric,
                args.inst_iou_thr,
                args.inst_min_area,
            )
            logger.info(
                "第 %d/%d 轮完成：训练损失 %.4f，像素准确率 %.4f，mIoU %.4f，mDice %.4f，用时 %.1f 秒",
                epoch,
                args.epochs,
                train_loss,
                acc,
                miou,
                mdice,
                time.time() - t0,
            )
            logger.info(
                "分类别指标：%s",
                " | ".join(
                    f"{name}: IoU {iou[i]:.4f}, Dice {dice[i]:.4f}"
                    for i, name in enumerate(class_names)
                ),
            )

            if inst is not None:
                logger.info(
                    "实例级指标：P %.4f，R %.4f，F1 %.4f，Acc %.4f，TP %d，FP %d，FN %d",
                    inst["precision"],
                    inst["recall"],
                    inst["f1"],
                    inst["acc"],
                    inst["tp"],
                    inst["fp"],
                    inst["fn"],
                )

            if miou > best_miou:
                best_miou = miou
                best_path = out_dir / "best_miou.pt"
                save_checkpoint(best_path, epoch, model, optimizer, scaler, class_names, best_miou, args)
                logger.info("检测到更优 mIoU，已保存最佳模型：%s", best_path)
        else:
            logger.info(
                "第 %d/%d 轮完成：训练损失 %.4f，用时 %.1f 秒（本轮未执行验证）",
                epoch,
                args.epochs,
                train_loss,
                time.time() - t0,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            # 周期性保存 checkpoint。
            ckpt_path = out_dir / f"segformer_epoch{epoch}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, class_names, best_miou, args)
            logger.info("已保存 checkpoint：%s", ckpt_path)

    logger.info("训练完成。")


if __name__ == "__main__":
    # 运行 pyhton
    main()

