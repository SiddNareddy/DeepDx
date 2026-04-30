#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from pytorch_model import (
    MURADataset,
    build_image_df,
    build_model,
    build_transforms,
    compute_study_metrics,
    predict,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic metal artifacts on normal MURA wrist images and evaluate checkpoints."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing MURA-v1.1 and model code.",
    )
    parser.add_argument(
        "--mura-root",
        type=Path,
        default=None,
        help="Path to MURA-v1.1 directory. Defaults to <workspace-root>/MURA-v1.1.",
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
        help="Directory containing one or more .pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated stress-test images. Defaults to <workspace-root>/stress_test_images.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <workspace-root>/stress_test_results.json.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Per-image prediction CSV path. Defaults to <workspace-root>/stress_test_predictions.csv.",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Directory for visualization outputs. Defaults to <workspace-root>/stress_test_visualizations.",
    )
    parser.add_argument("--num-images", type=int, default=10, help="Number of normal wrist images to patch.")
    parser.add_argument(
        "--num-viz-examples",
        type=int,
        default=6,
        help="How many original-vs-stress image pairs to render in the example grid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and patch placement.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device preference. 'auto' picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for inference.")
    parser.add_argument("--img-size", type=int, default=320, help="Resize size for validation transform.")
    parser.add_argument(
        "--artifact-style",
        choices=["block", "screw"],
        default="screw",
        help="Synthetic artifact shape.",
    )
    parser.add_argument("--patch-size", type=int, default=10, help="Square patch size in pixels.")
    parser.add_argument(
        "--screw-length",
        type=int,
        default=14,
        help="Length of the screw shaft when artifact-style=screw.",
    )
    parser.add_argument("--intensity", type=int, default=255, help="Pixel intensity used for artifact.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Abnormal threshold on sigmoid probability.")

    args = parser.parse_args()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.mura_root = (
        args.mura_root.expanduser().resolve()
        if args.mura_root is not None
        else args.workspace_root / "MURA-v1.1"
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.workspace_root / "stress_test_images"
    )
    args.results_path = (
        args.results_path.expanduser().resolve()
        if args.results_path is not None
        else args.workspace_root / "stress_test_results.json"
    )
    args.predictions_csv = (
        args.predictions_csv.expanduser().resolve()
        if args.predictions_csv is not None
        else args.workspace_root / "stress_test_predictions.csv"
    )
    args.viz_dir = (
        args.viz_dir.expanduser().resolve()
        if args.viz_dir is not None
        else args.workspace_root / "stress_test_visualizations"
    )
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if args.screw_length <= 0:
        raise ValueError("--screw-length must be > 0")
    if not (0 <= args.intensity <= 255):
        raise ValueError("--intensity must be in [0, 255]")
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.num_viz_examples <= 0:
        raise ValueError("--num-viz-examples must be > 0")
    if not args.mura_root.exists():
        raise FileNotFoundError(f"MURA root does not exist: {args.mura_root}")
    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Provide --checkpoint and/or --checkpoint-dir.")


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = []
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
        checkpoints.append(args.checkpoint)
    if args.checkpoint_dir is not None:
        if not args.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir does not exist: {args.checkpoint_dir}")
        dir_ckpts = sorted(args.checkpoint_dir.glob("*.pt"))
        checkpoints.extend(dir_ckpts)

    unique = []
    seen = set()
    for ckpt in checkpoints:
        ckpt_str = str(ckpt)
        if ckpt_str not in seen:
            seen.add(ckpt_str)
            unique.append(ckpt)
    if not unique:
        raise ValueError("No .pt checkpoints found from the provided arguments.")
    return unique


def overlay_block(arr: np.ndarray, rng: np.random.Generator, size: int, intensity: int) -> np.ndarray:
    h, w, _ = arr.shape
    size = min(size, h, w)
    y0 = int(rng.integers(0, h - size + 1))
    x0 = int(rng.integers(0, w - size + 1))
    arr[y0 : y0 + size, x0 : x0 + size, :] = intensity
    return arr


def overlay_screw(
    arr: np.ndarray,
    rng: np.random.Generator,
    patch_size: int,
    screw_length: int,
    intensity: int,
) -> np.ndarray:
    h, w, _ = arr.shape
    head_size = max(3, patch_size // 2)
    shaft_w = max(1, patch_size // 4)
    max_span = max(head_size, screw_length)
    if h < max_span or w < max_span:
        arr[:, :, :] = intensity
        return arr

    orientation = "vertical" if rng.random() < 0.5 else "horizontal"
    y0 = int(rng.integers(0, h - max_span + 1))
    x0 = int(rng.integers(0, w - max_span + 1))

    if orientation == "vertical":
        center_x = x0 + max_span // 2
        x1 = max(0, center_x - shaft_w // 2)
        x2 = min(w, x1 + shaft_w)
        y1 = y0
        y2 = min(h, y0 + screw_length)
        arr[y1:y2, x1:x2, :] = intensity
        hx0 = max(0, center_x - head_size // 2)
        hx1 = min(w, hx0 + head_size)
        hy0 = y0
        hy1 = min(h, y0 + head_size)
        arr[hy0:hy1, hx0:hx1, :] = intensity
    else:
        center_y = y0 + max_span // 2
        y1 = max(0, center_y - shaft_w // 2)
        y2 = min(h, y1 + shaft_w)
        x1 = x0
        x2 = min(w, x0 + screw_length)
        arr[y1:y2, x1:x2, :] = intensity
        hy0 = max(0, center_y - head_size // 2)
        hy1 = min(h, hy0 + head_size)
        hx0 = x0
        hx1 = min(w, x0 + head_size)
        arr[hy0:hy1, hx0:hx1, :] = intensity
    return arr


def generate_stress_images(
    selected_df: pd.DataFrame,
    workspace_root: Path,
    output_dir: Path,
    artifact_style: str,
    patch_size: int,
    screw_length: int,
    intensity: int,
    seed: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for idx, row in selected_df.reset_index(drop=True).iterrows():
        src = workspace_root / row["image_path"]
        if not src.exists():
            raise FileNotFoundError(f"Source image missing: {src}")

        image = Image.open(src).convert("RGB")
        arr = np.array(image, dtype=np.uint8)
        rng = np.random.default_rng(seed + idx)

        if artifact_style == "block":
            arr = overlay_block(arr, rng, patch_size, intensity)
        else:
            arr = overlay_screw(arr, rng, patch_size, screw_length, intensity)

        out_name = f"stress_{idx:03d}_{Path(row['image_path']).name}"
        out_path = output_dir / out_name
        Image.fromarray(arr).save(out_path)

        out_rows.append(
            {
                "image_path": str(out_path),
                "study_path": f"{row['study_path']}_stress_{idx:03d}",
                "split": "stress",
                "study_type": row["study_type"],
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "source_image_path": str(src),
                "generated_image_path": str(out_path),
            }
        )
    return pd.DataFrame(out_rows)


def build_loader(df: pd.DataFrame, data_root: Path, transform, batch_size: int, num_workers: int, device: torch.device):
    dataset = MURADataset(df, data_root, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    clean_loader: DataLoader,
    stress_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[dict, list[float], list[float]]:
    model = build_model(disable_pretrained=True).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)

    clean_probs, clean_labels, clean_study_paths, clean_study_types = predict(
        model,
        clean_loader,
        device,
        desc=f"clean [{checkpoint_path.name}]",
    )
    stress_probs, stress_labels, stress_study_paths, stress_study_types = predict(
        model,
        stress_loader,
        device,
        desc=f"stress [{checkpoint_path.name}]",
    )

    clean_metrics, _ = compute_study_metrics(
        clean_probs,
        clean_labels,
        clean_study_paths,
        clean_study_types,
        threshold=threshold,
    )
    stress_metrics, _ = compute_study_metrics(
        stress_probs,
        stress_labels,
        stress_study_paths,
        stress_study_types,
        threshold=threshold,
    )

    clean_pred = (np.asarray(clean_probs) >= threshold).astype(int)
    stress_pred = (np.asarray(stress_probs) >= threshold).astype(int)
    flip_to_abnormal = np.logical_and(clean_pred == 0, stress_pred == 1)

    result = {
        "checkpoint": str(checkpoint_path),
        "n_images": int(len(clean_probs)),
        "threshold": float(threshold),
        "clean_false_positive_rate": float(clean_pred.mean()),
        "stress_false_positive_rate": float(stress_pred.mean()),
        "flip_to_abnormal_count": int(flip_to_abnormal.sum()),
        "flip_to_abnormal_rate": float(flip_to_abnormal.mean()),
        "mean_clean_prob": float(np.mean(clean_probs)),
        "mean_stress_prob": float(np.mean(stress_probs)),
        "mean_prob_delta": float(np.mean(np.asarray(stress_probs) - np.asarray(clean_probs))),
        "clean_study_metrics": clean_metrics,
        "stress_study_metrics": stress_metrics,
    }
    return result, clean_probs, stress_probs


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def save_probability_shift_plot(
    df: pd.DataFrame,
    checkpoint_path: Path,
    threshold: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    clean_prob = df["clean_prob"].to_numpy(dtype=float)
    stress_prob = df["stress_prob"].to_numpy(dtype=float)
    delta_prob = df["delta_prob"].to_numpy(dtype=float)

    axes[0].scatter(clean_prob, stress_prob, alpha=0.85)
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].axvline(threshold, linestyle=":")
    axes[0].axhline(threshold, linestyle=":")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Clean probability")
    axes[0].set_ylabel("Stress probability")
    axes[0].set_title(f"Probabilities ({checkpoint_path.name})")

    axes[1].hist(delta_prob, bins=min(20, max(5, len(delta_prob))))
    axes[1].axvline(0.0, linestyle="--")
    axes[1].set_xlabel("Stress - clean probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction shift distribution")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_example_image_grid(
    stress_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    num_examples: int,
    output_path: Path,
    checkpoint_label: str,
) -> None:
    n = min(num_examples, len(stress_df))
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for row_idx in range(n):
        source_path = Path(stress_df.iloc[row_idx]["source_image_path"])
        generated_path = Path(stress_df.iloc[row_idx]["generated_image_path"])
        source_image = np.array(Image.open(source_path).convert("L"), dtype=np.uint8)
        generated_image = np.array(Image.open(generated_path).convert("L"), dtype=np.uint8)
        row_match = prediction_df[
            prediction_df["generated_image_path"] == str(generated_path)
        ]
        clean_prob = float(row_match.iloc[0]["clean_prob"]) if not row_match.empty else float("nan")
        stress_prob = float(row_match.iloc[0]["stress_prob"]) if not row_match.empty else float("nan")

        axes[row_idx, 0].imshow(source_image, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"Original #{row_idx + 1}\np={clean_prob:.3f}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(generated_image, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 1].set_title(f"Stress #{row_idx + 1}\np={stress_prob:.3f}")
        axes[row_idx, 1].axis("off")

    fig.suptitle(f"Stress examples with prediction probabilities ({Path(checkpoint_label).name})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_case_cell(ax, row: pd.Series | None, title: str) -> None:
    if row is None:
        ax.text(0.5, 0.5, "Not enough examples", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    image_path = Path(row["generated_image_path"])
    image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{title}\nprob={float(row['stress_prob']):.3f}")
    ax.axis("off")


def save_failure_case_grid(df: pd.DataFrame, output_path: Path) -> None:
    tn_df = df[df["stress_pred"] == 0].sort_values("stress_prob", ascending=True).head(2)
    fp_df = df[df["stress_pred"] == 1].sort_values("stress_prob", ascending=False).head(2)

    tn_rows = [tn_df.iloc[i] if i < len(tn_df) else None for i in range(2)]
    fp_rows = [fp_df.iloc[i] if i < len(fp_df) else None for i in range(2)]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    _render_case_cell(axes[0, 0], tn_rows[0], "True Negative #1")
    _render_case_cell(axes[0, 1], tn_rows[1], "True Negative #2")
    _render_case_cell(axes[1, 0], fp_rows[0], "False Positive #1")
    _render_case_cell(axes[1, 1], fp_rows[1], "False Positive #2")
    fig.suptitle("Failure Case Grid (Stress Images)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = select_device(args.device)
    checkpoints = collect_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    valid_images = build_image_df(args.mura_root, "valid")
    normal_wrist = valid_images[
        (valid_images["study_type"] == "XR_WRIST") & (valid_images["label"] == 0)
    ].reset_index(drop=True)
    if len(normal_wrist) < args.num_images:
        raise ValueError(
            f"Requested --num-images={args.num_images}, but only found {len(normal_wrist)} normal XR_WRIST images."
        )
    selected = normal_wrist.sample(n=args.num_images, random_state=args.seed).reset_index(drop=True)

    stress_df = generate_stress_images(
        selected_df=selected,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        artifact_style=args.artifact_style,
        patch_size=args.patch_size,
        screw_length=args.screw_length,
        intensity=args.intensity,
        seed=args.seed,
    )

    clean_df = selected.copy()
    clean_df = clean_df[
        ["image_path", "study_path", "split", "study_type", "patient_id", "label"]
    ].reset_index(drop=True)
    stress_dataset_df = stress_df[
        ["image_path", "study_path", "split", "study_type", "patient_id", "label"]
    ].reset_index(drop=True)

    _, val_transform = build_transforms(args.img_size, rotation_deg=0.0)
    clean_loader = build_loader(
        clean_df,
        data_root=args.workspace_root,
        transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    stress_loader = build_loader(
        stress_dataset_df,
        data_root=args.workspace_root,
        transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    run_results = []
    per_image_rows = []
    probability_plot_paths = []
    failure_case_grid_paths = []
    for checkpoint_path in checkpoints:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        summary, clean_probs, stress_probs = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            clean_loader=clean_loader,
            stress_loader=stress_loader,
            device=device,
            threshold=args.threshold,
        )
        run_results.append(summary)

        for i in range(len(clean_probs)):
            clean_prob = float(clean_probs[i])
            stress_prob = float(stress_probs[i])
            clean_pred = int(clean_prob >= args.threshold)
            stress_pred = int(stress_prob >= args.threshold)
            per_image_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "source_image_path": stress_df.loc[i, "source_image_path"],
                    "generated_image_path": stress_df.loc[i, "generated_image_path"],
                    "clean_prob": clean_prob,
                    "stress_prob": stress_prob,
                    "delta_prob": stress_prob - clean_prob,
                    "clean_pred": clean_pred,
                    "stress_pred": stress_pred,
                    "flip_to_abnormal": int(clean_pred == 0 and stress_pred == 1),
                }
            )

        print(
            f"  flip_to_abnormal={summary['flip_to_abnormal_count']}/{summary['n_images']} "
            f"({summary['flip_to_abnormal_rate']:.3f}) | "
            f"stress_FPR={summary['stress_false_positive_rate']:.3f}"
        )

        checkpoint_df = pd.DataFrame(
            [row for row in per_image_rows if row["checkpoint"] == str(checkpoint_path)]
        )
        prob_plot_path = args.viz_dir / f"probability_shift_{_safe_stem(checkpoint_path)}.png"
        save_probability_shift_plot(
            df=checkpoint_df,
            checkpoint_path=checkpoint_path,
            threshold=args.threshold,
            output_path=prob_plot_path,
        )
        probability_plot_paths.append(str(prob_plot_path))

        failure_grid_path = args.viz_dir / f"failure_case_grid_{_safe_stem(checkpoint_path)}.png"
        save_failure_case_grid(
            df=checkpoint_df,
            output_path=failure_grid_path,
        )
        failure_case_grid_paths.append(str(failure_grid_path))

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(args.predictions_csv, index=False)

    example_grid_checkpoint = str(checkpoints[0])
    example_grid_df = per_image_df[per_image_df["checkpoint"] == example_grid_checkpoint].copy()
    example_grid_path = args.viz_dir / "stress_examples_grid.png"
    save_example_image_grid(
        stress_df=stress_df,
        prediction_df=example_grid_df,
        num_examples=args.num_viz_examples,
        output_path=example_grid_path,
        checkpoint_label=example_grid_checkpoint,
    )

    output = {
        "config": {
            "workspace_root": str(args.workspace_root),
            "mura_root": str(args.mura_root),
            "output_dir": str(args.output_dir),
            "num_images": int(args.num_images),
            "seed": int(args.seed),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "img_size": int(args.img_size),
            "artifact_style": args.artifact_style,
            "patch_size": int(args.patch_size),
            "screw_length": int(args.screw_length),
            "intensity": int(args.intensity),
            "threshold": float(args.threshold),
            "checkpoints": [str(path) for path in checkpoints],
            "predictions_csv": str(args.predictions_csv),
            "viz_dir": str(args.viz_dir),
            "num_viz_examples": int(args.num_viz_examples),
        },
        "results": run_results,
        "visualizations": {
            "example_grid": str(example_grid_path),
            "example_grid_checkpoint": example_grid_checkpoint,
            "probability_shift_plots": probability_plot_paths,
            "failure_case_grids": failure_case_grid_paths,
        },
    }
    with args.results_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved per-image predictions to: {args.predictions_csv}")
    print(f"Saved stress test summary JSON to: {args.results_path}")
    print(f"Saved visualizations to: {args.viz_dir}")


if __name__ == "__main__":
    main()
