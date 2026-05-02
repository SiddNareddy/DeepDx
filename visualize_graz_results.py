#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = ["AUROC", "AUPRC", "sensitivity", "specificity", "precision", "f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GRAZPED evaluation summary results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("graz_ped_outputs/graz_ped_results.json"),
        help="Path to graz_ped_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <results_dir>/graz_ped_results_summary.png.",
    )
    return parser.parse_args()


def _first_run(results: dict) -> dict:
    runs = results.get("checkpoint_results", [])
    if not runs:
        raise ValueError("No checkpoint_results found in results JSON.")
    return runs[0]


def _confusion_matrix(metrics: dict) -> np.ndarray:
    return np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=int,
    )


def _plot_metric_bars(ax, image_metrics: dict, group_metrics: dict) -> None:
    x = np.arange(len(METRIC_NAMES))
    width = 0.38
    image_values = [image_metrics.get(name, np.nan) for name in METRIC_NAMES]
    group_values = [group_metrics.get(name, np.nan) for name in METRIC_NAMES]

    ax.bar(x - width / 2, image_values, width, label="image")
    ax.bar(x + width / 2, group_values, width, label="group")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)


def _plot_cohort(ax, counts: dict) -> None:
    labels = ["fracture", "control", "excluded"]
    values = [
        counts.get("positive_rows_fracture", 0),
        counts.get("control_rows", 0),
        counts.get("excluded_rows_non_control", 0),
    ]
    ax.bar(labels, values)
    ax.set_ylabel("Images")
    ax.set_title(f"Cohort Construction ({counts.get('control_policy', 'unknown')})")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)


def _plot_confusion(ax, matrix: np.ndarray, title: str) -> None:
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    results_path = args.results.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_path.parent / "graz_ped_results_summary.png"
    )

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    run = _first_run(results)
    image_metrics = run["image_metrics"]
    group_metrics = run["group_metrics"]
    counts = results.get("cohort_counts", {})
    threshold = results.get("threshold", image_metrics.get("threshold", 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _plot_cohort(axes[0, 0], counts)
    _plot_metric_bars(axes[0, 1], image_metrics, group_metrics)
    _plot_confusion(axes[1, 0], _confusion_matrix(image_metrics), "Image-Level Confusion Matrix")
    _plot_confusion(axes[1, 1], _confusion_matrix(group_metrics), "Group-Level Confusion Matrix")

    title = (
        f"GRAZPED Evaluation Summary: {run.get('checkpoint_name', 'checkpoint')} "
        f"(threshold={threshold}, model={run.get('model', 'unknown')})"
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
