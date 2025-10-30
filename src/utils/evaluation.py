"""
Evaluation metrics and utilities for GUI test automation
"""

import torch
from typing import List, Dict
import numpy as np
from collections import defaultdict
import re


try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: rouge-score or nltk not installed. Install with: pip install rouge-score nltk")


class GUITestEvaluator:
    """
    Evaluator for GUI test automation predictions.

    Metrics:
    1. Exact Match (EM): Percentage of exact matches
    2. BLEU: Measures n-gram overlap
    3. ROUGE-L: Measures longest common subsequence
    4. Action Type Accuracy: Accuracy on action type (click, type, etc.)
    """

    def __init__(self):
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.smooth = SmoothingFunction().method1
        else:
            self.rouge_scorer = None
            self.smooth = None

    def extract_action_type(self, text: str) -> str:
        """
        Extract action type from action text.

        Examples:
            "Click the login button" -> "click"
            "Type username into field" -> "type"
            "Verify results displayed" -> "verify"
        """
        text_lower = text.lower()

        # Common action types
        action_types = [
            'click', 'type', 'enter', 'select', 'scroll',
            'verify', 'check', 'wait', 'navigate', 'open',
            'close', 'submit', 'press', 'drag', 'hover'
        ]

        for action in action_types:
            if action in text_lower:
                return action

        return 'other'

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions) if predictions else 0.0

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate average BLEU score."""
        if not METRICS_AVAILABLE:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]  # BLEU expects list of references

            if pred_tokens and ref_tokens[0]:
                score = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    smoothing_function=self.smooth
                )
                scores.append(score)

        return np.mean(scores) if scores else 0.0

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not METRICS_AVAILABLE or self.rouge_scorer is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
        }

    def action_type_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate accuracy on action type (click, type, etc.)."""
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if self.extract_action_type(pred) == self.extract_action_type(ref)
        )
        return matches / len(predictions) if predictions else 0.0

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation.

        Args:
            predictions: List of predicted action texts
            references: List of ground truth action texts
            verbose: Print results

        Returns:
            Dict with all metrics
        """
        metrics = {}

        # Exact match
        metrics['exact_match'] = self.exact_match(predictions, references)

        # BLEU
        metrics['bleu'] = self.calculate_bleu(predictions, references)

        # ROUGE
        rouge_scores = self.calculate_rouge(predictions, references)
        metrics.update(rouge_scores)

        # Action type accuracy
        metrics['action_type_acc'] = self.action_type_accuracy(predictions, references)

        if verbose:
            print("\n" + "="*60)
            print("Evaluation Results")
            print("="*60)
            for key, value in metrics.items():
                print(f"{key:20s}: {value:.4f} ({value*100:.2f}%)")
            print("="*60 + "\n")

        return metrics

    def evaluate_by_workflow(
        self,
        predictions: List[str],
        references: List[str],
        workflows: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate separately for each workflow.

        Args:
            predictions: Predicted actions
            references: Ground truth actions
            workflows: Workflow identifiers

        Returns:
            Dict mapping workflow -> metrics
        """
        # Group by workflow
        workflow_data = defaultdict(lambda: {'predictions': [], 'references': []})

        for pred, ref, wf in zip(predictions, references, workflows):
            workflow_data[wf]['predictions'].append(pred)
            workflow_data[wf]['references'].append(ref)

        # Evaluate each workflow
        results = {}
        for workflow, data in workflow_data.items():
            results[workflow] = self.evaluate(
                data['predictions'],
                data['references'],
                verbose=False
            )

        return results


@torch.no_grad()
def evaluate_model(model, dataloader, evaluator=None, max_batches=None):
    """
    Evaluate model on a dataset.

    Args:
        model: GUITestBLIP2 model
        dataloader: Validation/test dataloader
        evaluator: GUITestEvaluator instance
        max_batches: Maximum batches to evaluate (for quick testing)

    Returns:
        Dict with metrics
    """
    if evaluator is None:
        evaluator = GUITestEvaluator()

    model.model.eval()

    all_predictions = []
    all_references = []
    all_workflows = []

    print("Generating predictions...")
    from tqdm.auto import tqdm

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if max_batches and batch_idx >= max_batches:
            break

        # Get images and prompts
        images = [dataloader.dataset.dataset[idx]['image']
                  for idx in range(batch_idx * dataloader.batch_size,
                                  min((batch_idx + 1) * dataloader.batch_size,
                                      len(dataloader.dataset.dataset)))]

        # Reconstruct input texts (prompts)
        input_texts = []
        for i in range(len(images)):
            idx = batch_idx * dataloader.batch_size + i
            if idx >= len(dataloader.dataset.dataset):
                break
            sample = dataloader.dataset.dataset[idx]
            input_texts.append(dataloader.dataset._construct_input_text(sample))

        # Generate predictions
        predictions = model.generate(
            images=images[:len(input_texts)],
            prompts=input_texts,
            max_length=128,
            num_beams=4,
        )

        # Get references
        references = batch['target_texts']

        # Get workflows
        workflows = batch['workflows']

        all_predictions.extend(predictions)
        all_references.extend(references)
        all_workflows.extend(workflows)

    # Evaluate
    print(f"\nEvaluating {len(all_predictions)} predictions...")
    metrics = evaluator.evaluate(all_predictions, all_references)

    # Per-workflow metrics
    print("\nPer-workflow results:")
    workflow_metrics = evaluator.evaluate_by_workflow(
        all_predictions,
        all_references,
        all_workflows
    )

    for workflow, wf_metrics in workflow_metrics.items():
        print(f"\n{workflow}:")
        for key, value in wf_metrics.items():
            print(f"  {key}: {value:.4f}")

    return {
        'overall': metrics,
        'per_workflow': workflow_metrics,
        'predictions': all_predictions,
        'references': all_references,
    }


if __name__ == "__main__":
    # Test evaluator
    evaluator = GUITestEvaluator()

    # Example predictions and references
    predictions = [
        "Click the login button",
        "Type username into field",
        "Click submit",
    ]

    references = [
        "Click the login button",
        "Enter username in the text field",
        "Click the submit button",
    ]

    metrics = evaluator.evaluate(predictions, references)
    print("\nTest completed successfully!")
