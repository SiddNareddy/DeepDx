import argparse
import json
import os
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from pytorch_alt_model import (
    MODEL_CHOICES,
    build_model,
    build_transforms,
    choose_num_workers,
    predict,
    select_device,
)


FRACTURE_CLASS_ID = 3
GRAZ_STUDY_TYPE = "GRAZPEDWRI_DX"
CONTROL_POLICIES = [
    "empty-labels",
    "no-fracture",
    "no-fracture-text-ok",
    "strict-clean",
]
NON_TEXT_PATHOLOGIC_CLASSES = {0, 1, 2, 4, 5, 6, 7}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MURA-trained checkpoints on GRAZPEDWRI-DX with configurable control policies."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root path.",
    )
    parser.add_argument(
        "--graz-root",
        type=Path,
        default=None,
        help="Path to GRAZPEDWRI-DX root. Defaults to <workspace-root>/GRAZPEDWRI-DX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a single .pt checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory with .pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to <workspace-root>/graz_ped_outputs.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="JSON summary path. Defaults to <output-dir>/graz_ped_results.json.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Per-image probabilities CSV. Defaults to <output-dir>/graz_ped_predictions.csv.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Base plot path. Additional per-checkpoint plots will be suffixed.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device preference. 'auto' picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. Defaults to the same policy as pytorch_alt_model.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override validation resize size. If omitted, uses checkpoint config.img_size or 320.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary predictions from abnormal probability.",
    )
    parser.add_argument(
        "--model",
        choices=["auto"] + MODEL_CHOICES,
        default="auto",
        help="Model architecture to instantiate. 'auto' uses checkpoint config.model or densenet169.",
    )
    parser.add_argument(
        "--control-policy",
        choices=CONTROL_POLICIES,
        default="no-fracture-text-ok",
        help=(
            "How to define control images: "
            "empty-labels (strict), no-fracture (fracture_visible is missing), "
            "no-fracture-text-ok (no fracture and only text/empty boxes), "
            "strict-clean (no fracture + text/empty boxes + metadata flags missing)."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of evaluation images for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for optional subsampling.")

    args = parser.parse_args()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.graz_root = (
        args.graz_root.expanduser().resolve()
        if args.graz_root is not None
        else (args.workspace_root / "GRAZPEDWRI-DX").resolve()
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (args.workspace_root / "graz_ped_outputs").resolve()
    )
    args.results_path = (
        args.results_path.expanduser().resolve()
        if args.results_path is not None
        else (args.output_dir / "graz_ped_results.json").resolve()
    )
    args.predictions_csv = (
        args.predictions_csv.expanduser().resolve()
        if args.predictions_csv is not None
        else (args.output_dir / "graz_ped_predictions.csv").resolve()
    )
    args.plot_path = (
        args.plot_path.expanduser().resolve()
        if args.plot_path is not None
        else (args.output_dir / "graz_ped_probability_scores.png").resolve()
    )
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.graz_root.exists():
        raise FileNotFoundError(f"GRAZPEDWRI-DX root does not exist: {args.graz_root}")
    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Provide --checkpoint and/or --checkpoint-dir.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be > 0 when provided.")


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = []
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
        checkpoints.append(args.checkpoint)
    if args.checkpoint_dir is not None:
        if not args.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
        checkpoints.extend(sorted(args.checkpoint_dir.glob("*.pt")))

    unique = []
    seen: set[str] = set()
    for ckpt in checkpoints:
        key = str(ckpt)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ckpt)
    if not unique:
        raise ValueError("No checkpoints found. Check --checkpoint/--checkpoint-dir inputs.")
    return unique


def index_image_paths(graz_root: Path) -> dict[str, Path]:
    parts = [graz_root / f"images_part{i}" for i in range(1, 5)]
    mapping: dict[str, Path] = {}
    for part_dir in parts:
        if not part_dir.exists():
            continue
        for image_path in sorted(part_dir.glob("*.png")):
            mapping[image_path.stem] = image_path.resolve()
    return mapping


def parse_yolo_label_file(label_path: Path) -> tuple[bool, bool, int, set[int]]:
    if not label_path.exists():
        return False, False, 0, set()
    text = label_path.read_text(encoding="utf-8").strip()
    if text == "":
        return True, False, 0, set()

    has_fracture = False
    non_empty = 0
    class_ids: set[int] = set()
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        non_empty += 1
        first = row.split()[0]
        if first.isdigit():
            class_id = int(first)
            class_ids.add(class_id)
            if class_id == FRACTURE_CLASS_ID:
                has_fracture = True
    return False, has_fracture, non_empty, class_ids


def _is_missing(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _strict_clean_flags_missing(row: pd.Series) -> bool:
    keys = ["ao_classification", "diagnosis_uncertain", "cast", "metal", "osteopenia"]
    return all(_is_missing(row.get(key)) for key in keys)


def _is_control_by_policy(
    *,
    policy: str,
    fracture_visible_value,
    is_empty_label: bool,
    class_ids: set[int],
    strict_clean_flags_missing: bool,
) -> bool:
    no_fracture_visible = _is_missing(fracture_visible_value)
    text_only_or_empty = (len(class_ids) == 0) or class_ids.issubset({8})
    if policy == "empty-labels":
        return is_empty_label
    if policy == "no-fracture":
        return no_fracture_visible
    if policy == "no-fracture-text-ok":
        return no_fracture_visible and text_only_or_empty
    if policy == "strict-clean":
        return no_fracture_visible and text_only_or_empty and strict_clean_flags_missing
    raise ValueError(f"Unknown control policy: {policy}")


def build_eval_dataframe(graz_root: Path, control_policy: str) -> tuple[pd.DataFrame, dict]:
    dataset_csv = graz_root / "dataset.csv"
    labels_dir = graz_root / "folder_structure" / "yolov5" / "labels"
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {dataset_csv}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    image_map = index_image_paths(graz_root)
    raw_df = pd.read_csv(dataset_csv)
    rows = []
    excluded_non_control = 0
    missing_images = 0
    missing_labels = 0
    no_fracture_visible_rows = 0
    text_only_or_empty_rows = 0
    strict_clean_candidate_rows = 0

    total_rows = len(raw_df)
    progress_interval = 500
    for idx, (_, row) in enumerate(raw_df.iterrows(), start=1):
        if idx == 1 or idx % progress_interval == 0 or idx == total_rows:
            print(
                f"  scanned {idx}/{total_rows} dataset rows; "
                f"included={len(rows)}, missing_images={missing_images}, missing_labels={missing_labels}",
                flush=True,
            )

        filestem = str(row["filestem"])
        image_path = image_map.get(filestem)
        if image_path is None:
            missing_images += 1
            continue

        label_path = labels_dir / f"{filestem}.txt"
        is_empty, has_fracture, n_boxes, class_ids = parse_yolo_label_file(label_path)
        if not label_path.exists():
            missing_labels += 1
            continue

        fracture_visible_value = row.get("fracture_visible", np.nan)
        strict_flags_missing = _strict_clean_flags_missing(row)
        text_only_or_empty = (len(class_ids) == 0) or class_ids.issubset({8})
        has_non_text_pathologic = any(cls in NON_TEXT_PATHOLOGIC_CLASSES for cls in class_ids)
        no_fracture_visible = _is_missing(fracture_visible_value)

        if no_fracture_visible:
            no_fracture_visible_rows += 1
        if text_only_or_empty:
            text_only_or_empty_rows += 1
        if no_fracture_visible and text_only_or_empty and strict_flags_missing:
            strict_clean_candidate_rows += 1

        if has_fracture:
            binary_label = 1
        elif _is_control_by_policy(
            policy=control_policy,
            fracture_visible_value=fracture_visible_value,
            is_empty_label=is_empty,
            class_ids=class_ids,
            strict_clean_flags_missing=strict_flags_missing,
        ):
            binary_label = 0
        else:
            excluded_non_control += 1
            continue

        patient_id = str(row.get("patient_id", ""))
        study_number = str(row.get("study_number", ""))
        laterality = str(row.get("laterality", ""))
        group_id = f"patient_{patient_id}_study_{study_number}_lat_{laterality}"
        projection = row.get("projection", np.nan)
        view_id = f"{filestem}_proj{projection}"

        rows.append(
            {
                "filestem": filestem,
                "image_path": str(image_path),
                "label": int(binary_label),
                "group_id": group_id,
                "study_type": GRAZ_STUDY_TYPE,
                "patient_id": patient_id,
                "study_number": study_number,
                "laterality": laterality,
                "projection": projection,
                "view_id": view_id,
                "n_boxes": int(n_boxes),
                "is_empty_label": bool(is_empty),
                "has_fracture_box": bool(has_fracture),
                "has_non_text_pathologic_box": bool(has_non_text_pathologic),
            }
        )

    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        raise RuntimeError(
            f"No images available for fracture-vs-control evaluation using policy '{control_policy}'."
        )

    counts = {
        "control_policy": control_policy,
        "raw_dataset_rows": int(len(raw_df)),
        "included_rows": int(len(eval_df)),
        "positive_rows_fracture": int((eval_df["label"] == 1).sum()),
        "control_rows": int((eval_df["label"] == 0).sum()),
        "excluded_rows_non_control": int(excluded_non_control),
        "missing_images": int(missing_images),
        "missing_labels": int(missing_labels),
        "dataset_rows_no_fracture_visible": int(no_fracture_visible_rows),
        "dataset_rows_text_only_or_empty_labels": int(text_only_or_empty_rows),
        "dataset_rows_strict_clean_candidates": int(strict_clean_candidate_rows),
    }
    return eval_df.reset_index(drop=True), counts


class GRAZPEDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, float(row["label"]), row["study_type"], row["group_id"]


def build_loader(
    df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    _, val_transform = build_transforms(img_size=img_size, rotation_deg=0.0)
    dataset = GRAZPEDDataset(df, transform=val_transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]
    return {
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "AUROC": _safe_auc(y_true, y_score),
        "AUPRC": _safe_auprc(y_true, y_score),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "accuracy": acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_abnormal_prob_positive": float(np.mean(positive_scores)) if len(positive_scores) else float("nan"),
        "mean_abnormal_prob_negative": float(np.mean(negative_scores)) if len(negative_scores) else float("nan"),
        "median_abnormal_prob_positive": float(np.median(positive_scores)) if len(positive_scores) else float("nan"),
        "median_abnormal_prob_negative": float(np.median(negative_scores)) if len(negative_scores) else float("nan"),
        "threshold": float(threshold),
    }


def _metrics_with_groups(
    base_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> tuple[dict, dict]:
    image_true = base_df["label"].to_numpy(dtype=int)
    image_metrics = compute_binary_metrics(image_true, probs, threshold=threshold)

    group_df = base_df[["group_id", "label"]].copy()
    group_df["prob"] = probs
    grouped = (
        group_df.groupby("group_id")
        .agg(group_prob=("prob", "mean"), group_label=("label", "first"))
        .reset_index()
    )
    group_true = grouped["group_label"].to_numpy(dtype=int)
    group_prob = grouped["group_prob"].to_numpy(dtype=float)
    grouped_metrics = compute_binary_metrics(group_true, group_prob, threshold=threshold)
    grouped_metrics["n_groups"] = int(len(grouped))
    return image_metrics, grouped_metrics


def resolve_checkpoint_config(
    payload: object,
    model_override: str,
    img_size_override: int | None,
) -> tuple[str, int, dict]:
    payload_dict = payload if isinstance(payload, dict) else {}
    config = payload_dict.get("config", {}) if isinstance(payload_dict, dict) else {}
    model_name = model_override if model_override != "auto" else config.get("model", "densenet169")
    if model_name not in MODEL_CHOICES:
        raise ValueError(
            f"Unsupported model '{model_name}'. Expected one of {MODEL_CHOICES}. "
            "Pass --model to override."
        )
    img_size = int(img_size_override) if img_size_override is not None else int(config.get("img_size", 320))
    return model_name, img_size, config


def _state_from_payload(payload: object) -> dict:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint payload is not a dictionary-like state dict.")


def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    num_workers: int,
) -> tuple[dict, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device)
    model_name, img_size, config = resolve_checkpoint_config(payload, args.model, args.img_size)
    model = build_model(disable_pretrained=True, model_name=model_name).to(device)
    model.load_state_dict(_state_from_payload(payload))

    loader = build_loader(
        df=eval_df,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
        device=device,
    )
    probs, labels, _, _ = predict(model, loader, device, desc=f"graz [{checkpoint_path.name}]")
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    if len(probs_arr) != len(eval_df):
        raise RuntimeError(
            f"Prediction length mismatch for {checkpoint_path.name}: got {len(probs_arr)}, expected {len(eval_df)}"
        )
    if not np.array_equal(labels_arr, eval_df["label"].to_numpy(dtype=int)):
        raise RuntimeError("Label mismatch between loader output and evaluation dataframe.")

    image_metrics, grouped_metrics = _metrics_with_groups(eval_df, probs_arr, threshold=args.threshold)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "model": model_name,
        "img_size": int(img_size),
        "threshold": float(args.threshold),
        "image_metrics": image_metrics,
        "group_metrics": grouped_metrics,
        "checkpoint_config": config,
    }
    return result, probs_arr


def evaluate_ensemble(
    eval_df: pd.DataFrame,
    probs_by_checkpoint: list[np.ndarray],
    threshold: float,
) -> tuple[dict, np.ndarray]:
    matrix = np.vstack(probs_by_checkpoint)
    ensemble_probs = np.mean(matrix, axis=0)
    image_metrics, grouped_metrics = _metrics_with_groups(eval_df, ensemble_probs, threshold=threshold)
    result = {
        "checkpoint": "ensemble_mean",
        "checkpoint_name": "ensemble_mean",
        "image_metrics": image_metrics,
        "group_metrics": grouped_metrics,
        "n_members": int(matrix.shape[0]),
        "threshold": float(threshold),
    }
    return result, ensemble_probs


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def plot_probability_scores(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    output_path: Path,
    title: str,
    control_label: str,
) -> None:
    neg = df[df["label"] == 0][score_col].to_numpy(dtype=float)
    pos = df[df["label"] == 1][score_col].to_numpy(dtype=float)
    bins = np.linspace(0.0, 1.0, 41)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(neg, bins=bins, alpha=0.6, label=control_label, density=True)
    axes[0].hist(pos, bins=bins, alpha=0.6, label="fracture", density=True)
    axes[0].axvline(threshold, linestyle="--", linewidth=1.2, label="threshold")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Abnormal probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Probability Distribution")
    axes[0].legend()

    axes[1].boxplot([neg, pos], labels=["control", "fracture"], showfliers=False)
    axes[1].axhline(threshold, linestyle="--", linewidth=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Abnormal probability")
    axes[1].set_title("Score Spread")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str,
    run_label: str,
    auroc_value: float,
    auprc_value: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    has_both_classes = len(np.unique(y_true)) > 1
    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{run_label} (AUROC={auroc_value:.4f})")
        axes[1].plot(recall, precision, label=f"{run_label} (AUPRC={auprc_value:.4f})")
    else:
        axes[0].text(0.5, 0.5, "ROC unavailable\n(single class in labels)", ha="center", va="center")
        axes[1].text(0.5, 0.5, "PR unavailable\n(single class in labels)", ha="center", va="center")

    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="chance")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    baseline = float(np.mean(y_true))
    axes[1].axhline(baseline, linestyle="--", linewidth=1.0, label=f"prevalence={baseline:.4f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_comparison(
    runs: list[dict],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="chance")

    for run in runs:
        y_true = run["y_true"]
        y_score = run["y_score"]
        if len(np.unique(y_true)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{run['name']} ({run['auroc']:.4f})")
        axes[1].plot(recall, precision, label=f"{run['name']} ({run['auprc']:.4f})")

    if runs:
        baseline = float(np.mean(runs[0]["y_true"]))
        axes[1].axhline(baseline, linestyle="--", linewidth=1.0, label=f"prevalence={baseline:.4f}")

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC Comparison")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Comparison")
    axes[1].legend(loc="lower left", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_json_ready(value):
    if isinstance(value, dict):
        return {k: _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = select_device(args.device)
    num_workers = choose_num_workers(device, args.num_workers)
    checkpoints = collect_checkpoints(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {args.control_policy} policy")

    eval_df, cohort_counts = build_eval_dataframe(args.graz_root, args.control_policy)
    if args.max_images is not None:
        eval_df = (
            eval_df.sample(n=min(args.max_images, len(eval_df)), random_state=args.seed)
            .sort_values("filestem")
            .reset_index(drop=True)
        )

    print("Configuration:")
    print(f"  graz_root      : {args.graz_root}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  results_path   : {args.results_path}")
    print(f"  predictions_csv: {args.predictions_csv}")
    print(f"  plot_path      : {args.plot_path}")
    print(f"  device         : {device}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  num_workers    : {num_workers}")
    print(f"  threshold      : {args.threshold}")
    print(f"  model override : {args.model}")
    print(f"  control_policy : {args.control_policy}")
    print(f"  img_size override: {args.img_size}")
    print(f"  checkpoints    : {len(checkpoints)}")

    print("Cohort counts:")
    for k, v in cohort_counts.items():
        print(f"  {k:32s} {v}")
    print(f"  final_eval_rows{'':20s} {len(eval_df)}")
    print(eval_df["label"].value_counts().rename(index={0: "control", 1: "fracture"}).to_string())

    per_image_df = eval_df[
        [
            "filestem",
            "image_path",
            "label",
            "group_id",
            "study_type",
            "patient_id",
            "study_number",
            "laterality",
            "projection",
            "view_id",
            "n_boxes",
        ]
    ].copy()

    control_label_by_policy = {
        "empty-labels": "control (empty-label only)",
        "no-fracture": "control (no visible fracture)",
        "no-fracture-text-ok": "control (no fracture, text/empty labels)",
        "strict-clean": "control (strict clean)",
    }
    control_label = control_label_by_policy[args.control_policy]

    checkpoint_results = []
    probs_for_ensemble: list[np.ndarray] = []
    roc_pr_plot_paths: list[dict[str, str]] = []
    curve_runs: list[dict] = []
    y_true = eval_df["label"].to_numpy(dtype=int)
    for checkpoint_path in checkpoints:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        result, probs = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            eval_df=eval_df,
            args=args,
            device=device,
            num_workers=num_workers,
        )
        checkpoint_results.append(result)
        probs_for_ensemble.append(probs)

        col = f"abnormal_prob__{_safe_name(checkpoint_path)}"
        per_image_df[col] = probs
        plot_file = args.plot_path.with_name(f"{args.plot_path.stem}_{_safe_name(checkpoint_path)}.png")
        plot_probability_scores(
            df=per_image_df.rename(columns={col: "abnormal_prob"}),
            score_col="abnormal_prob",
            threshold=args.threshold,
            output_path=plot_file,
            title=f"GRAZPED abnormal probabilities: {checkpoint_path.name}",
            control_label=control_label,
        )
        roc_pr_file = args.plot_path.with_name(f"{args.plot_path.stem}_{_safe_name(checkpoint_path)}_roc_pr.png")
        plot_roc_pr_curves(
            y_true=y_true,
            y_score=probs,
            output_path=roc_pr_file,
            title=f"GRAZPED ROC/PR curves: {checkpoint_path.name}",
            run_label=checkpoint_path.name,
            auroc_value=result["image_metrics"]["AUROC"],
            auprc_value=result["image_metrics"]["AUPRC"],
        )
        roc_pr_plot_paths.append({"run": checkpoint_path.name, "path": str(roc_pr_file)})
        curve_runs.append(
            {
                "name": checkpoint_path.name,
                "y_true": y_true,
                "y_score": probs,
                "auroc": float(result["image_metrics"]["AUROC"]),
                "auprc": float(result["image_metrics"]["AUPRC"]),
            }
        )

    ensemble_result = None
    if len(probs_for_ensemble) > 1:
        ensemble_result, ensemble_probs = evaluate_ensemble(
            eval_df=eval_df,
            probs_by_checkpoint=probs_for_ensemble,
            threshold=args.threshold,
        )
        per_image_df["abnormal_prob__ensemble_mean"] = ensemble_probs
        ensemble_plot = args.plot_path.with_name(f"{args.plot_path.stem}_ensemble_mean.png")
        plot_probability_scores(
            df=per_image_df.rename(columns={"abnormal_prob__ensemble_mean": "abnormal_prob"}),
            score_col="abnormal_prob",
            threshold=args.threshold,
            output_path=ensemble_plot,
            title="GRAZPED abnormal probabilities: ensemble mean",
            control_label=control_label,
        )
        ensemble_curve_plot = args.plot_path.with_name(f"{args.plot_path.stem}_ensemble_mean_roc_pr.png")
        plot_roc_pr_curves(
            y_true=y_true,
            y_score=ensemble_probs,
            output_path=ensemble_curve_plot,
            title="GRAZPED ROC/PR curves: ensemble mean",
            run_label="ensemble_mean",
            auroc_value=ensemble_result["image_metrics"]["AUROC"],
            auprc_value=ensemble_result["image_metrics"]["AUPRC"],
        )
        roc_pr_plot_paths.append({"run": "ensemble_mean", "path": str(ensemble_curve_plot)})
        curve_runs.append(
            {
                "name": "ensemble_mean",
                "y_true": y_true,
                "y_score": ensemble_probs,
                "auroc": float(ensemble_result["image_metrics"]["AUROC"]),
                "auprc": float(ensemble_result["image_metrics"]["AUPRC"]),
            }
        )

    roc_pr_comparison_plot = None
    if len(curve_runs) > 1:
        roc_pr_comparison_path = args.plot_path.with_name(f"{args.plot_path.stem}_roc_pr_comparison.png")
        plot_roc_pr_comparison(
            runs=curve_runs,
            output_path=roc_pr_comparison_path,
            title="GRAZPED ROC/PR comparison across runs",
        )
        roc_pr_comparison_plot = str(roc_pr_comparison_path)

    long_rows = []
    for col in [c for c in per_image_df.columns if c.startswith("abnormal_prob__")]:
        run_name = col.replace("abnormal_prob__", "")
        run_prob = per_image_df[col].to_numpy(dtype=float)
        run_pred = (run_prob >= args.threshold).astype(int)
        for i in range(len(per_image_df)):
            long_rows.append(
                {
                    "run": run_name,
                    "filestem": per_image_df.loc[i, "filestem"],
                    "label": int(per_image_df.loc[i, "label"]),
                    "group_id": per_image_df.loc[i, "group_id"],
                    "abnormal_prob": float(run_prob[i]),
                    "pred_label": int(run_pred[i]),
                    "image_path": per_image_df.loc[i, "image_path"],
                }
            )
    pred_long_df = pd.DataFrame(long_rows)
    pred_long_df.to_csv(args.predictions_csv, index=False)

    results_payload = {
        "graz_root": str(args.graz_root),
        "cohort_counts": cohort_counts,
        "final_eval_rows": int(len(eval_df)),
        "threshold": float(args.threshold),
        "control_policy": args.control_policy,
        "checkpoints": [str(p) for p in checkpoints],
        "checkpoint_results": checkpoint_results,
        "ensemble_result": ensemble_result,
        "predictions_csv": str(args.predictions_csv),
        "plot_base_path": str(args.plot_path),
        "roc_pr_plot_paths": roc_pr_plot_paths,
        "roc_pr_comparison_plot": roc_pr_comparison_plot,
    }
    args.results_path.write_text(json.dumps(_to_json_ready(results_payload), indent=2), encoding="utf-8")

    print(f"Saved results JSON: {args.results_path}")
    print(f"Saved predictions CSV: {args.predictions_csv}")
    print(f"Saved plots with base: {args.plot_path}")
    if roc_pr_plot_paths:
        print("Saved ROC/PR plots:")
        for item in roc_pr_plot_paths:
            print(f"  {item['run']}: {item['path']}")
    if roc_pr_comparison_plot:
        print(f"Saved ROC/PR comparison plot: {roc_pr_comparison_plot}")


if __name__ == "__main__":
    main()
