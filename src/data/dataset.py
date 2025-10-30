"""
GUIDE Dataset Loader for GUI Test Automation
Loads and preprocesses the SuperAGI/GUIDE dataset
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import json
from typing import Dict, List, Optional


class GUITestDataset(Dataset):
    """
    Dataset for GUI test automation using GUIDE dataset.

    Each sample contains:
    - image: Screenshot of the application
    - question: Task description
    - previousAction: Last action taken
    - previousActionHistory: List of all previous actions
    - answer: Next action to predict (target)
    - cot: Chain of thought reasoning
    """

    def __init__(
        self,
        split: str = "train",
        max_history_length: int = 10,
        max_seq_length: int = 512,
        use_cot: bool = False,
    ):
        """
        Args:
            split: 'train', 'validation', or 'test'
            max_history_length: Maximum number of previous actions to include
            max_seq_length: Maximum sequence length for text
            use_cot: Whether to include chain-of-thought in training
        """
        self.split = split
        self.max_history_length = max_history_length
        self.max_seq_length = max_seq_length
        self.use_cot = use_cot

        print(f"Loading GUIDE dataset ({split} split)...")
        # Load from HuggingFace
        self.dataset = load_dataset("SuperAGI/GUIDE", split=split)
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def _format_history(self, history) -> str:
        """Format action history into readable text."""
        import re

        # Handle None or empty
        if not history or history == [None] or history == 'None':
            return "None"

        # Handle string (multi-line string with numbered actions)
        if isinstance(history, str):
            # Split by newlines to get individual actions
            lines = [line.strip() for line in history.split('\n') if line.strip()]

            if lines:
                # Remove existing numbering (e.g., "1. CLICK: ..." -> "CLICK: ...")
                actions = []
                for line in lines:
                    # Match patterns like "1. ", "1) ", "• ", etc.
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if cleaned:  # Only add non-empty actions
                        actions.append(cleaned)

                history = actions if actions else [history]
            else:
                # Single action without newlines
                history = [history]

        # Handle list (already in list format)
        if isinstance(history, list):
            # Filter out None/empty values
            history = [h for h in history if h and str(h).strip() and str(h).strip() != 'None']

            if not history:
                return "None"

            # Take last N actions (most recent)
            history = history[-self.max_history_length:]

            # Format with numbering
            formatted = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])
            return formatted

        # Fallback
        return "None"

    def _construct_input_text(self, item: Dict) -> str:
        """
        Construct input text from question, history, and current action.

        Format:
        Task: {question}
        Previous steps:
        1. {action1}
        2. {action2}
        Current action: {previousAction}
        Next action:
        """
        question = item.get('question', 'No task description')
        previous_action = item.get('previousAction', 'None')
        history = item.get('previousActionHistory', [])

        # Handle None values
        if previous_action is None or previous_action == 'None':
            previous_action = "None (starting new workflow)"

        history_text = self._format_history(history)

        input_text = f"""Task: {question}

Previous steps:
{history_text}

Current action: {previous_action}

Predict the next action:"""

        return input_text

    def _get_target_text(self, item: Dict) -> str:
        """Get target text (answer or answer + CoT)."""
        answer = item.get('answer', '')

        if self.use_cot:
            cot = item.get('cot', '')
            if cot:
                return f"Reasoning: {cot}\nAction: {answer}"

        return answer

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - image: PIL Image
                - input_text: Formatted input prompt
                - target_text: Expected output (next action)
                - workflow: Workflow identifier
                - raw_data: Original dataset item
        """
        item = self.dataset[idx]

        # Load image
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')

        # Construct input and target
        input_text = self._construct_input_text(item)
        target_text = self._get_target_text(item)

        return {
            'image': image,
            'input_text': input_text,
            'target_text': target_text,
            'workflow': item.get('workflow', 'unknown'),
            'raw_data': item
        }


class GUITestCollator:
    """
    Collator for batching GUI test samples.
    Works with BLIP-2 processor.
    """

    def __init__(self, processor, device='cuda'):
        """
        Args:
            processor: BLIP-2 processor (handles image + text preprocessing)
            device: Device to move tensors to
        """
        self.processor = processor
        self.device = device

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.

        Args:
            batch: List of samples from GUITestDataset

        Returns:
            Dict with processed inputs ready for model
        """
        images = [item['image'] for item in batch]
        input_texts = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]

        # Process images and text together
        encoding = self.processor(
            images=images,
            text=input_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Process targets (for labels)
        labels = self.processor.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).input_ids

        # Replace padding token id with -100 (ignore in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        labels = labels.to(self.device)

        return {
            **encoding,
            'labels': labels,
            'target_texts': target_texts,  # Keep for evaluation
            'workflows': [item['workflow'] for item in batch]
        }


def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 2,
    processor=None,
    device='cuda',
    max_history_length: int = 10,
    use_cot: bool = False,
    val_split_ratio: float = 0.1
):
    """
    Create train and validation dataloaders.

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        processor: BLIP-2 processor
        device: Device for tensors
        max_history_length: Max previous actions to include
        use_cot: Include chain-of-thought
        val_split_ratio: Ratio of training data to use for validation

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    import torch

    # Load full training dataset
    full_dataset = GUITestDataset(
        split='train',
        max_history_length=max_history_length,
        use_cot=use_cot
    )

    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    print(f"\nSplitting dataset:")
    print(f"  Total samples: {total_size}")
    print(f"  Training: {train_size} ({(1-val_split_ratio)*100:.0f}%)")
    print(f"  Validation: {val_size} ({val_split_ratio*100:.0f}%)")

    # Set seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create collator
    collator = GUITestCollator(processor, device)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if device == 'cuda' else False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    dataset = GUITestDataset(split='train')
    print(f"\nDataset size: {len(dataset)}")

    # Show example
    sample = dataset[0]
    print(f"\nExample sample:")
    print(f"Image size: {sample['image'].size}")
    print(f"Input text:\n{sample['input_text']}")
    print(f"\nTarget text: {sample['target_text']}")
    print(f"Workflow: {sample['workflow']}")
