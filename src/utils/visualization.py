"""
Visualization utilities for GUI test automation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict
import seaborn as sns


def plot_sample(sample: Dict, figsize=(12, 8)):
    """
    Plot a single dataset sample with all information.

    Args:
        sample: Sample dict from GUITestDataset
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Display image
    ax1.imshow(sample['image'])
    ax1.axis('off')
    ax1.set_title('Screenshot', fontsize=14, fontweight='bold')

    # Display text information
    text_content = f"""WORKFLOW: {sample['workflow']}

INPUT PROMPT:
{sample['input_text']}

TARGET ACTION:
{sample['target_text']}
"""

    ax2.text(0.05, 0.95, text_content,
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.axis('off')

    plt.tight_layout()
    return fig


def plot_predictions_comparison(
    images: List[Image.Image],
    predictions: List[str],
    ground_truths: List[str],
    tasks: List[str] = None,
    n_samples: int = 3,
    figsize=(15, 5)
):
    """
    Plot predictions vs ground truth for multiple samples.

    Args:
        images: List of PIL Images
        predictions: List of predicted actions
        ground_truths: List of ground truth actions
        tasks: List of task descriptions (optional)
        n_samples: Number of samples to plot
        figsize: Figure size per sample
    """
    n_samples = min(n_samples, len(images))

    fig, axes = plt.subplots(n_samples, 2, figsize=(figsize[0], figsize[1] * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Display image
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')

        # Check if match
        is_match = predictions[i].lower().strip() == ground_truths[i].lower().strip()
        match_color = 'lightgreen' if is_match else 'lightcoral'
        match_emoji = '✓' if is_match else '✗'

        # Display comparison
        task_text = f"Task: {tasks[i][:80]}...\n\n" if tasks and i < len(tasks) else ""
        text_content = f"""{task_text}PREDICTED {match_emoji}:
{predictions[i]}

GROUND TRUTH:
{ground_truths[i]}

EXACT MATCH: {is_match}
"""

        axes[i, 1].text(0.05, 0.95, text_content,
                       transform=axes[i, 1].transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor=match_color, alpha=0.6))
        axes[i, 1].axis('off')

    plt.tight_layout()
    return fig


def plot_sequence_generation(
    image: Image.Image,
    task: str,
    sequence: List[Dict],
    figsize=(14, 10)
):
    """
    Visualize generated test sequence.

    Args:
        image: Screenshot image
        task: Task description
        sequence: List of step dicts from generate_sequence()
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # Top: Image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(f'Task: {task}', fontsize=14, fontweight='bold', pad=20)

    # Bottom: Sequence
    ax_seq = fig.add_subplot(gs[1])
    ax_seq.axis('off')

    # Format sequence text
    sequence_text = "GENERATED TEST SEQUENCE:\n" + "="*60 + "\n\n"
    for step in sequence:
        sequence_text += f"Step {step['step_num']}: {step['action']}\n\n"
    sequence_text += "="*60

    ax_seq.text(0.05, 0.95, sequence_text,
               transform=ax_seq.transAxes,
               fontsize=11,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, float],
    title: str = "Model Performance Metrics",
    figsize=(10, 6)
):
    """
    Plot evaluation metrics as a bar chart.

    Args:
        metrics_dict: Dict of metric_name -> score
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    metrics = list(metrics_dict.keys())
    scores = [metrics_dict[m] * 100 for m in metrics]  # Convert to percentage

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
    bars = ax.barh(metrics, scores, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{width:.2f}%',
               ha='left', va='center', fontweight='bold')

    ax.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_workflow_metrics(
    workflow_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'exact_match',
    figsize=(12, 6)
):
    """
    Plot metrics across different workflows.

    Args:
        workflow_metrics: Dict of workflow -> metrics dict
        metric_name: Which metric to plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    workflows = list(workflow_metrics.keys())
    scores = [workflow_metrics[wf].get(metric_name, 0) * 100 for wf in workflows]

    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]
    workflows = [workflows[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    colors = plt.cm.coolwarm(np.array(scores) / 100)
    bars = ax.bar(range(len(workflows)), scores, color=colors,
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{score:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(workflows)))
    ax.set_xticklabels(workflows, rotation=45, ha='right')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name.replace("_", " ").title()} by Workflow',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    figsize=(12, 5)
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss',
            linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(epochs, val_losses, 's-', label='Val Loss',
            linewidth=2, markersize=8, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    ax2.bar(epochs, loss_diff, color='#F77F00', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|Train Loss - Val Loss|', fontsize=12, fontweight='bold')
    ax2.set_title('Overfitting Check', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_action_type_distribution(
    actions: List[str],
    title: str = "Action Type Distribution",
    figsize=(10, 6)
):
    """
    Plot distribution of action types (click, type, verify, etc.).

    Args:
        actions: List of action text strings
        title: Plot title
        figsize: Figure size
    """
    from collections import Counter
    from utils.evaluation import GUITestEvaluator

    evaluator = GUITestEvaluator()
    action_types = [evaluator.extract_action_type(action) for action in actions]
    counts = Counter(action_types)

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(counts.keys())
    values = list(counts.values())

    # Sort by count
    sorted_idx = np.argsort(values)[::-1]
    labels = [labels[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Action Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - plot_sample()")
    print("  - plot_predictions_comparison()")
    print("  - plot_sequence_generation()")
    print("  - plot_metrics_comparison()")
    print("  - plot_workflow_metrics()")
    print("  - plot_training_curves()")
    print("  - create_action_type_distribution()")
