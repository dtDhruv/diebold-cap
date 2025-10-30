"""
BLIP-2 Model for GUI Test Automation
Fine-tuned BLIP-2 with FLAN-T5 for generating test steps
"""

import torch
import torch.nn as nn
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoProcessor
)
from typing import Dict, List, Optional, Tuple


class GUITestBLIP2(nn.Module):
    """
    BLIP-2 model fine-tuned for GUI test automation.

    Architecture:
        - Vision Encoder: EVA-CLIP ViT-g (frozen)
        - Q-Former: Querying Transformer (trainable) - compresses visual info
        - Language Model: FLAN-T5 (trainable decoder, frozen encoder)

    This efficiently bridges vision and language by:
    1. Extracting visual features with frozen CLIP
    2. Compressing them with Q-Former (32 tokens)
    3. Feeding to T5 for text generation
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-base",
        device: str = "cuda",
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        freeze_lm_encoder: bool = True,
    ):
        """
        Args:
            model_name: Pre-trained BLIP-2 model name
            device: Device to load model on
            freeze_vision: Keep vision encoder frozen (recommended)
            freeze_qformer: Freeze Q-Former (not recommended, we want to train this)
            freeze_lm_encoder: Freeze T5 encoder (recommended)
        """
        super().__init__()

        print(f"Loading BLIP-2 model: {model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.device = device
        self.model.to(device)

        # Load processor (handles image + text preprocessing)
        self.processor = Blip2Processor.from_pretrained(model_name)

        # Configure which parts to train
        self._configure_trainable_params(
            freeze_vision, freeze_qformer, freeze_lm_encoder
        )

        print(f"Model loaded on {device}")
        self._print_trainable_params()

    def _configure_trainable_params(
        self, freeze_vision: bool, freeze_qformer: bool, freeze_lm_encoder: bool
    ):
        """Configure which model components are trainable."""

        # 1. Vision encoder (CLIP ViT) - Usually frozen
        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            print("✓ Vision encoder frozen")
        else:
            print("✓ Vision encoder trainable")

        # 2. Q-Former - We want to train this!
        if freeze_qformer:
            for param in self.model.qformer.parameters():
                param.requires_grad = False
            print("✓ Q-Former frozen")
        else:
            print("✓ Q-Former trainable")

        # 3. Language model (T5)
        if freeze_lm_encoder:
            # Freeze encoder, train decoder
            for param in self.model.language_model.encoder.parameters():
                param.requires_grad = False
            print("✓ T5 encoder frozen, decoder trainable")
        else:
            print("✓ Full T5 trainable")

    def _print_trainable_params(self):
        """Print number of trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass.

        Args:
            pixel_values: Image tensors [B, 3, H, W]
            input_ids: Text input IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Target labels [B, target_len] (for training)

        Returns:
            Dict with loss, logits, etc.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        images: List,
        prompts: List[str],
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate next actions for given images and prompts.

        Args:
            images: List of PIL Images
            prompts: List of text prompts
            max_length: Maximum generation length
            num_beams: Beam search width (higher = better quality, slower)
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling threshold

        Returns:
            List of generated action texts
        """
        self.model.eval()

        # Preprocess inputs
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        # Decode
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_texts

    def generate_sequence(
        self,
        initial_image,
        question: str,
        max_steps: int = 10,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> List[Dict]:
        """
        Generate a full sequence of test steps autoregressively.

        Args:
            initial_image: Starting screenshot (PIL Image)
            question: Task description
            max_steps: Maximum number of steps to generate
            max_length: Max length per action
            num_beams: Beam search width

        Returns:
            List of dicts with:
                - step_num: Step number
                - action: Generated action text
                - prompt: Input prompt used
        """
        self.model.eval()

        sequence = []
        history = []
        current_image = initial_image

        for step in range(max_steps):
            # Construct prompt with current history
            history_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])
            if not history_text:
                history_text = "None"

            prompt = f"""Task: {question}

Previous steps:
{history_text}

Current action: {history[-1] if history else 'None'}

Predict the next action:"""

            # Generate next action
            next_actions = self.generate(
                images=[current_image],
                prompts=[prompt],
                max_length=max_length,
                num_beams=num_beams,
            )
            next_action = next_actions[0]

            # Check for completion
            if "done" in next_action.lower() or "complete" in next_action.lower():
                break

            # Add to sequence
            sequence.append({
                "step_num": step + 1,
                "action": next_action,
                "prompt": prompt,
            })

            # Update history
            history.append(next_action)

            # In real scenario, would update current_image based on action
            # For now, keep same image (dataset limitation)

        return sequence

    def save_model(self, save_path: str):
        """Save model and processor."""
        print(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print("✓ Model saved")

    def load_model(self, load_path: str):
        """Load model and processor."""
        print(f"Loading model from {load_path}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(load_path)
        self.processor = Blip2Processor.from_pretrained(load_path)
        self.model.to(self.device)
        print("✓ Model loaded")


def load_pretrained_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> GUITestBLIP2:
    """
    Load pre-trained or fine-tuned model.

    Args:
        checkpoint_path: Path to fine-tuned checkpoint (None for base model)
        device: Device to load on

    Returns:
        GUITestBLIP2 model
    """
    if checkpoint_path:
        print(f"Loading fine-tuned model from {checkpoint_path}")
        model = GUITestBLIP2(model_name=checkpoint_path, device=device)
    else:
        print("Loading base BLIP-2 model")
        model = GUITestBLIP2(device=device)

    return model


if __name__ == "__main__":
    # Test model loading
    print("Testing BLIP-2 model...")
    model = GUITestBLIP2(device="cuda" if torch.cuda.is_available() else "cpu")

    # Test generation
    from PIL import Image
    import requests
    from io import BytesIO

    # Load test image
    url = "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400"
    response = requests.get(url)
    test_image = Image.open(BytesIO(response.content))

    prompt = "Task: Click the login button\nPrevious steps:\nNone\nPredict the next action:"

    print("\nGenerating test prediction...")
    result = model.generate([test_image], [prompt], max_length=50)
    print(f"Generated: {result[0]}")
