#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())


STUDY_TYPES = ["XR_WRIST", "XR_SHOULDER"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ADAM_BETAS = (0.9, 0.999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DenseNet-169 on the wrist+shoulder MURA subset."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing MURA-v1.1 and output folders.",
    )
    parser.add_argument(
        "--mura-root",
        type=Path,
        default=None,
        help="Path to the MURA-v1.1 directory. Defaults to <workspace-root>/MURA-v1.1.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints. Defaults to <workspace-root>/checkpoints.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="JSON path for training results. Defaults to <workspace-root>/training_results.json.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="PNG path for training curves. Defaults to <workspace-root>/training_curves.png.",
    )
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=20)
    #parser.add_argument("--lr-patience", type=int, default=1)
    #parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--rotation-deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device preference. 'auto' picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to SLURM_CPUS_PER_TASK-1 when available.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end pass for quick verification.",
    )
    parser.add_argument("--smoke-test-images", type=int, default=16)
    parser.add_argument("--smoke-test-epochs", type=int, default=1)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of lowest-validation-loss checkpoints to retain and ensemble.",
    )
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the training curves after saving them.",
    )
    args = parser.parse_args()

    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.mura_root = (
        args.mura_root.expanduser().resolve()
        if args.mura_root
        else args.workspace_root / "MURA-v1.1"
    )
    args.checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir
        else args.workspace_root / "checkpoints"
    )
    args.results_path = (
        args.results_path.expanduser().resolve()
        if args.results_path
        else args.workspace_root / "training_results.json"
    )
    args.plot_path = (
        args.plot_path.expanduser().resolve()
        if args.plot_path
        else args.workspace_root / "training_curves.png"
    )
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("--device mps requested but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_num_workers(device: torch.device, requested: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    if device.type == "mps":
        return 0
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus) - 1)
    return 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_paths(args: argparse.Namespace) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.mura_root.exists():
        raise FileNotFoundError(f"MURA root does not exist: {args.mura_root}")


def build_image_df(mura_root: Path, split: str) -> pd.DataFrame:
    csv_path = mura_root / f"{split}_image_paths.csv"
    df = pd.read_csv(csv_path, header=None, names=["image_path"])
    df["image_path"] = df["image_path"].str.strip()
    parts = df["image_path"].str.split("/", expand=True)
    df["study_path"] = df["image_path"].str.rsplit("/", n=1).str[0]
    df["split"] = parts[1]
    df["study_type"] = parts[2]
    df["patient_id"] = parts[3]
    df["label"] = parts[4].str.contains("_positive").astype(np.int64)
    return df[["image_path", "study_path", "split", "study_type", "patient_id", "label"]]


def maybe_smoke_test(
    train_images: pd.DataFrame,
    valid_images: pd.DataFrame,
    valid_studies: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not args.smoke_test:
        return train_images, valid_images, valid_studies
    train_images = train_images.sample(
        n=min(args.smoke_test_images, len(train_images)),
        random_state=args.seed,
    ).reset_index(drop=True)
    keep_valid = valid_studies.sample(
        n=min(64, len(valid_studies)),
        random_state=args.seed,
    )["study_path"]
    valid_images = valid_images[
        valid_images["study_path"].isin(keep_valid)
    ].reset_index(drop=True)
    valid_studies = valid_studies[
        valid_studies["study_path"].isin(keep_valid)
    ].reset_index(drop=True)
    print("SMOKE_TEST active:")
    print(f"  train images: {len(train_images)}  valid images: {len(valid_images)}")
    return train_images, valid_images, valid_studies


def compute_class_weights(image_df: pd.DataFrame) -> dict:
    weights = {}
    for study_type in STUDY_TYPES:
        sub = image_df[image_df["study_type"] == study_type]
        n_abnormal = int((sub["label"] == 1).sum())
        n_normal = int((sub["label"] == 0).sum())
        total = n_abnormal + n_normal
        if total == 0:
            raise ValueError(f"No training images for study type {study_type}")
        weights[study_type] = {
            "w_1": n_normal / total,
            "w_0": n_abnormal / total,
            "n_abnormal": n_abnormal,
            "n_normal": n_normal,
        }
    return weights


class MURADataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.data_root / row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, float(row["label"]), row["study_type"], row["study_path"]


class WeightedBCEByStudyType(nn.Module):
    def __init__(self, weights_by_type: dict):
        super().__init__()
        self.weights_by_type = weights_by_type

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, study_types):
        logits = logits.squeeze(-1)
        labels = labels.float()
        w1 = torch.tensor(
            [self.weights_by_type[study_type]["w_1"] for study_type in study_types],
            device=logits.device,
            dtype=logits.dtype,
        )
        w0 = torch.tensor(
            [self.weights_by_type[study_type]["w_0"] for study_type in study_types],
            device=logits.device,
            dtype=logits.dtype,
        )
        sample_weights = labels * w1 + (1.0 - labels) * w0
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            weight=sample_weights,
            reduction="mean",
        )


def build_transforms(img_size: int, rotation_deg: float):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_deg, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_model(disable_pretrained: bool) -> nn.Module:
    weights = None if disable_pretrained else models.DenseNet169_Weights.IMAGENET1K_V1
    model = models.densenet169(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def _progress_bar(loader, desc):
    return tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, desc="train"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = _progress_bar(loader, desc)
    for images, labels, study_types, _ in pbar:
        t_batch = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels, list(study_types))
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        loss_val = loss.item()
        total_loss += loss_val * bs
        total_samples += bs
        batch_ms = (time.time() - t_batch) * 1000.0
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                avg=f"{total_loss / max(total_samples, 1):.4f}",
                bt=f"{batch_ms:.0f}ms",
            )
    if hasattr(pbar, "close"):
        pbar.close()
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, desc="valid"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probs_out, labels_out, sp_out, st_out = [], [], [], []
    pbar = _progress_bar(loader, desc)
    for images, labels, study_types, study_paths in pbar:
        t_batch = time.time()
        images = images.to(device, non_blocking=True)
        labels_d = labels.float().to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels_d, list(study_types))
        bs = images.size(0)
        loss_val = loss.item()
        total_loss += loss_val * bs
        total_samples += bs
        probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
        probs_out.extend(probs.tolist())
        labels_out.extend(labels.numpy().tolist())
        sp_out.extend(list(study_paths))
        st_out.extend(list(study_types))
        batch_ms = (time.time() - t_batch) * 1000.0
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                avg=f"{total_loss / max(total_samples, 1):.4f}",
                bt=f"{batch_ms:.0f}ms",
            )
    if hasattr(pbar, "close"):
        pbar.close()
    return total_loss / max(total_samples, 1), probs_out, labels_out, sp_out, st_out


def compute_study_metrics(probs, labels, study_paths, study_types, threshold: float = 0.5):
    df = pd.DataFrame(
        {
            "prob": probs,
            "label": labels,
            "study_path": study_paths,
            "study_type": study_types,
        }
    )
    study_df = (
        df.groupby("study_path")
        .agg(
            study_prob=("prob", "mean"),
            study_label=("label", "first"),
            study_type=("study_type", "first"),
            n_views=("prob", "size"),
        )
        .reset_index()
    )
    study_df["study_pred"] = (study_df["study_prob"] >= threshold).astype(int)

    metrics = {}
    groups = [("overall", study_df)] + [
        (study_type, study_df[study_df["study_type"] == study_type])
        for study_type in STUDY_TYPES
    ]
    for name, sub in groups:
        if len(sub) == 0:
            continue
        y_true = sub["study_label"].values.astype(int)
        y_score = sub["study_prob"].values.astype(float)
        y_pred = sub["study_pred"].values.astype(int)
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auroc = float("nan")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            kappa = float("nan")
        metrics[name] = {
            "AUROC": auroc,
            "sensitivity": sens,
            "specificity": spec,
            "cohen_kappa": kappa,
            "n_studies": int(len(sub)),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
        }
    return metrics, study_df


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


SUMMARY_METRIC_COLS = ["AUROC", "sensitivity", "specificity", "cohen_kappa"]
SUMMARY_COUNT_COLS = ["n_studies", "n_positive", "n_negative"]
SUMMARY_GROUP_ORDER = ["overall"] + STUDY_TYPES


class TopKCheckpointTracker:
    """Retain the K checkpoints with the lowest validation losses on disk."""

    def __init__(self, checkpoint_dir: Path, top_k: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.top_k = max(1, int(top_k))
        self.entries: list[dict] = []

    def consider(self, *, epoch: int, val_loss: float, payload: dict) -> Path | None:
        if len(self.entries) >= self.top_k:
            worst = max(self.entries, key=lambda e: e["val_loss"])
            if val_loss >= worst["val_loss"]:
                return None
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.6f}.pt"
        torch.save(payload, path)
        self.entries.append({"epoch": int(epoch), "val_loss": float(val_loss), "path": path})
        self.entries.sort(key=lambda e: e["val_loss"])
        while len(self.entries) > self.top_k:
            removed = self.entries.pop()
            try:
                Path(removed["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        return path

    def best(self) -> dict | None:
        return self.entries[0] if self.entries else None

    def manifest(self) -> list[dict]:
        return [
            {
                "epoch": entry["epoch"],
                "val_loss": entry["val_loss"],
                "path": str(entry["path"]),
            }
            for entry in self.entries
        ]


@torch.no_grad()
def predict(model, loader, device, desc: str = "predict"):
    """Run inference and return per-image probabilities and identifiers."""
    model.eval()
    probs_out, labels_out, sp_out, st_out = [], [], [], []
    pbar = _progress_bar(loader, desc)
    for images, labels, study_types, study_paths in pbar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
        probs_out.extend(probs.tolist())
        labels_out.extend(labels.numpy().tolist())
        sp_out.extend(list(study_paths))
        st_out.extend(list(study_types))
    if hasattr(pbar, "close"):
        pbar.close()
    return probs_out, labels_out, sp_out, st_out


def ensemble_predict(model, checkpoint_entries, loader, device):
    """Sequentially load each checkpoint and average per-image probabilities."""
    if not checkpoint_entries:
        raise ValueError("ensemble_predict requires at least one checkpoint entry")
    accumulator = None
    labels_ref = None
    sp_ref = None
    st_ref = None
    for idx, entry in enumerate(checkpoint_entries, start=1):
        ckpt = torch.load(entry["path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        desc = f"ensemble {idx}/{len(checkpoint_entries)} (ep {entry['epoch']:02d})"
        probs, labels_, sp_, st_ = predict(model, loader, device, desc=desc)
        probs_arr = np.asarray(probs, dtype=np.float64)
        if accumulator is None:
            accumulator = probs_arr.copy()
            labels_ref = labels_
            sp_ref = sp_
            st_ref = st_
        else:
            if len(probs_arr) != len(accumulator):
                raise RuntimeError(
                    "Ensemble probability vectors have mismatched lengths; "
                    "verify the validation loader is deterministic."
                )
            accumulator += probs_arr
    avg_probs = (accumulator / len(checkpoint_entries)).tolist()
    return avg_probs, labels_ref, sp_ref, st_ref


def build_summary_table(final_metrics: dict) -> pd.DataFrame:
    columns = ["group"] + SUMMARY_METRIC_COLS + SUMMARY_COUNT_COLS
    rows = []
    for name in SUMMARY_GROUP_ORDER:
        if name not in final_metrics:
            continue
        m = final_metrics[name]
        row = {"group": name}
        for col in SUMMARY_METRIC_COLS + SUMMARY_COUNT_COLS:
            row[col] = m.get(col)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def print_summary_table(df: pd.DataFrame, title: str = "Final summary") -> None:
    if df.empty:
        print(f"{title}: <empty>")
        return
    formatted = df.copy()
    for col in SUMMARY_METRIC_COLS:
        formatted[col] = formatted[col].map(
            lambda v: "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        )
    for col in SUMMARY_COUNT_COLS:
        formatted[col] = formatted[col].map(
            lambda v: "" if v is None else f"{int(v)}"
        )
    print(title)
    print(formatted.to_string(index=False))


def save_plots(history: dict, plot_path: Path, show_plot: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = list(range(1, len(history["train_loss"]) + 1))

    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("weighted BCE loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    overall_auroc = [
        metric.get("overall", {}).get("AUROC", float("nan"))
        for metric in history["study_metrics"]
    ]
    wrist_auroc = [
        metric.get("XR_WRIST", {}).get("AUROC", float("nan"))
        for metric in history["study_metrics"]
    ]
    shoulder_auroc = [
        metric.get("XR_SHOULDER", {}).get("AUROC", float("nan"))
        for metric in history["study_metrics"]
    ]
    axes[1].plot(epochs_range, overall_auroc, label="overall")
    axes[1].plot(epochs_range, wrist_auroc, label="XR_WRIST")
    axes[1].plot(epochs_range, shoulder_auroc, label="XR_SHOULDER")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("study AUROC")
    axes[1].set_title("Validation AUROC (study level)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"saved plots to {plot_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_paths(args)
    device = select_device(args.device)
    num_workers = choose_num_workers(device, args.num_workers)
    pin_memory = device.type == "cuda"
    data_root = args.workspace_root

    seed_everything(args.seed)

    print("Configuration:")
    print(f"  workspace_root : {args.workspace_root}")
    print(f"  mura_root      : {args.mura_root}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  results_path   : {args.results_path}")
    print(f"  plot_path      : {args.plot_path}")
    print(f"  device         : {device}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  num_workers    : {num_workers}")
    print(f"  epochs         : {args.smoke_test_epochs if args.smoke_test else args.num_epochs}")
    print(f"  pretrained     : {not args.disable_pretrained}")

    train_images = build_image_df(args.mura_root, "train")
    valid_images = build_image_df(args.mura_root, "valid")
    train_studies = train_images[
        ["study_path", "study_type", "label", "patient_id"]
    ].drop_duplicates(subset="study_path").reset_index(drop=True)
    valid_studies = valid_images[
        ["study_path", "study_type", "label", "patient_id"]
    ].drop_duplicates(subset="study_path").reset_index(drop=True)

    print(f"train images : {len(train_images):6d}  studies: {len(train_studies):5d}")
    print(f"valid images : {len(valid_images):6d}  studies: {len(valid_studies):5d}")
    print("Training image counts by study_type x label")
    print(train_images.groupby(["study_type", "label"]).size().unstack(fill_value=0))
    print()
    print("Validation study counts by study_type x label")
    print(valid_studies.groupby(["study_type", "label"]).size().unstack(fill_value=0))

    train_images, valid_images, valid_studies = maybe_smoke_test(
        train_images,
        valid_images,
        valid_studies,
        args,
    )

    class_weights = compute_class_weights(train_images)
    for study_type, weights in class_weights.items():
        print(
            f"{study_type:12s}  n_abnormal={weights['n_abnormal']:5d}  "
            f"n_normal={weights['n_normal']:5d}  "
            f"w_1={weights['w_1']:.4f}  w_0={weights['w_0']:.4f}"
        )

    train_transform, val_transform = build_transforms(args.img_size, args.rotation_deg)
    train_dataset = MURADataset(train_images, data_root, train_transform)
    valid_dataset = MURADataset(valid_images, data_root, val_transform)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"train batches : {len(train_loader):5d}  valid batches : {len(valid_loader):5d}")
    imgs, labels, study_types, study_paths = next(iter(valid_loader))
    print("image batch shape :", tuple(imgs.shape))
    print("label batch       :", labels[:8].tolist())
    print("study_types       :", list(study_types)[:4])
    print("study_paths[0]    :", study_paths[0])

    model = build_model(args.disable_pretrained).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_trainable / 1e6:.2f} M")

    criterion = WeightedBCEByStudyType(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=ADAM_BETAS)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
     #   optimizer,
     #   mode="min",
     #   factor=args.lr_factor,
     #   patience=args.lr_patience,
    #)

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "study_metrics": [],
        "epoch_time_s": [],
        "train_time_s": [],
        "val_time_s": [],
    }
    epochs_to_run = args.smoke_test_epochs if args.smoke_test else args.num_epochs
    ensemble_size = max(1, min(args.ensemble_size, epochs_to_run))
    tracker = TopKCheckpointTracker(args.checkpoint_dir, ensemble_size)
    run_started = time.time()

    print(
        f"training for {epochs_to_run} epoch(s) | "
        f"train batches/epoch={len(train_loader)} | "
        f"valid batches/epoch={len(valid_loader)} | "
        f"batch_size={args.batch_size} | device={device}"
    )

    for epoch in range(1, epochs_to_run + 1):
        print(f"\\n=== epoch {epoch:02d}/{epochs_to_run} ===")
        t_epoch = time.time()

        t_train_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"ep {epoch:02d}/{epochs_to_run} [train]",
        )
        train_time = time.time() - t_train_start

        t_val_start = time.time()
        val_loss, probs, labels_, study_paths, study_types = validate(
            model,
            valid_loader,
            criterion,
            device,
            desc=f"ep {epoch:02d}/{epochs_to_run} [valid]",
        )
        val_time = time.time() - t_val_start

        study_metrics, _ = compute_study_metrics(
            probs,
            labels_,
            study_paths,
            study_types,
        )
        #scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        epoch_elapsed = time.time() - t_epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)
        history["study_metrics"].append(study_metrics)
        history["epoch_time_s"].append(epoch_elapsed)
        history["train_time_s"].append(train_time)
        history["val_time_s"].append(val_time)

        avg_epoch_s = sum(history["epoch_time_s"]) / len(history["epoch_time_s"])
        epochs_left = epochs_to_run - epoch
        eta_s = avg_epoch_s * epochs_left
        total_elapsed = time.time() - run_started
        overall = study_metrics.get("overall", {})

        print(
            f"[ep {epoch:02d}/{epochs_to_run}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={lr_now:.1e} | "
            f"AUROC={overall.get('AUROC', float('nan')):.4f} | "
            f"sens={overall.get('sensitivity', float('nan')):.4f} | "
            f"spec={overall.get('specificity', float('nan')):.4f}"
        )
        print(
            f"  times: epoch={_fmt_hms(epoch_elapsed)} "
            f"(train {_fmt_hms(train_time)}, valid {_fmt_hms(val_time)})  |  "
            f"total={_fmt_hms(total_elapsed)}  "
            f"avg/epoch={_fmt_hms(avg_epoch_s)}  "
            f"ETA({epochs_left} left)={_fmt_hms(eta_s)}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": val_loss,
            "study_metrics": study_metrics,
            "class_weights": class_weights,
            "config": {
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "study_types": STUDY_TYPES,
                "seed": args.seed,
            },
        }
        saved_path = tracker.consider(epoch=epoch, val_loss=val_loss, payload=ckpt_payload)
        if saved_path is not None:
            print(
                f"  -> retained checkpoint ({saved_path.name}, val_loss={val_loss:.4f}); "
                f"top-{tracker.top_k} size={len(tracker.entries)}"
            )

    print(f"\\nTraining finished in {_fmt_hms(time.time() - run_started)}")

    if not tracker.entries:
        raise RuntimeError("No checkpoints were retained; training may have failed")

    print(f"\\nRetained top-{tracker.top_k} checkpoints (lowest validation loss first):")
    for rank, entry in enumerate(tracker.entries, start=1):
        print(
            f"  [{rank}] epoch={entry['epoch']:02d}  val_loss={entry['val_loss']:.4f}  "
            f"path={Path(entry['path']).name}"
        )

    best_entry = tracker.best()
    print(
        f"\\nLoading single best checkpoint (epoch {best_entry['epoch']}, "
        f"val_loss={best_entry['val_loss']:.4f}) for comparison"
    )
    best_ckpt = torch.load(best_entry["path"], map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    best_probs, best_labels, best_sp, best_st = predict(
        model,
        valid_loader,
        device,
        desc="single-best [valid]",
    )
    best_metrics, _ = compute_study_metrics(best_probs, best_labels, best_sp, best_st)

    print(
        f"\\nRunning ensemble evaluation across {len(tracker.entries)} checkpoint(s)"
    )
    ens_probs, ens_labels, ens_sp, ens_st = ensemble_predict(
        model,
        tracker.entries,
        valid_loader,
        device,
    )
    ensemble_metrics, ensemble_study_df = compute_study_metrics(
        ens_probs, ens_labels, ens_sp, ens_st
    )

    best_table = build_summary_table(best_metrics)
    ensemble_table = build_summary_table(ensemble_metrics)
    print()
    print_summary_table(
        best_table,
        title=f"Single best checkpoint (epoch {best_entry['epoch']}) summary:",
    )
    print()
    print_summary_table(
        ensemble_table,
        title=f"Ensemble of top-{len(tracker.entries)} checkpoints summary:",
    )

    final_metrics = ensemble_metrics

    results = {
        "history": history,
        "final_metrics": final_metrics,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_summary_table": ensemble_table.to_dict(orient="records"),
        "best_metrics": best_metrics,
        "best_summary_table": best_table.to_dict(orient="records"),
        "best_epoch": int(best_entry["epoch"]),
        "best_val_loss": float(best_entry["val_loss"]),
        "ensemble_size": len(tracker.entries),
        "ensemble_checkpoints": tracker.manifest(),
        "class_weights": class_weights,
        "config": {
            "workspace_root": str(args.workspace_root),
            "mura_root": str(args.mura_root),
            "checkpoint_dir": str(args.checkpoint_dir),
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": epochs_to_run,
            #"lr_patience": args.lr_patience,
            #"lr_factor": args.lr_factor,
            "rotation_deg": args.rotation_deg,
            "study_types": STUDY_TYPES,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "num_workers": num_workers,
            "device": str(device),
            "pretrained": not args.disable_pretrained,
            "ensemble_size_requested": args.ensemble_size,
        },
    }
    with args.results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\\nsaved results to {args.results_path}")

    save_plots(history, args.plot_path, args.show_plot)


if __name__ == "__main__":
    main()
