"""
Inference script for GUI Test Automation
Run predictions on new screenshots

Usage:
    python inference.py --image screenshot.png --task "Login to app"
    python inference.py --image screenshot.png --task "Search for product" --history "Open homepage" "Click search"
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

sys.path.append('./src')

from models.blip2_model import GUITestBLIP2


def format_history(history_list):
    """Format history list into numbered text."""
    if not history_list:
        return "None"
    return "\n".join([f"{i+1}. {action}" for i, action in enumerate(history_list)])


def predict_single_step(model, image_path, task, history=None, current_action=None):
    """
    Predict next single action.

    Args:
        model: Trained model
        image_path: Path to screenshot
        task: Task description
        history: List of previous actions
        current_action: Current action (last action taken)

    Returns:
        Predicted next action
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Format history
    history_text = format_history(history) if history else "None"
    current = current_action if current_action else "None"

    # Create prompt
    prompt = f"""Task: {task}

Previous steps:
{history_text}

Current action: {current}

Predict the next action:"""

    print("\n" + "="*60)
    print("INPUT")
    print("="*60)
    print(prompt)

    # Generate prediction
    print("\n" + "="*60)
    print("GENERATING...")
    print("="*60)

    prediction = model.generate(
        images=[image],
        prompts=[prompt],
        max_length=128,
        num_beams=4,
        temperature=0.7
    )[0]

    print("\n" + "="*60)
    print("PREDICTED NEXT ACTION")
    print("="*60)
    print(f"→ {prediction}")
    print("="*60 + "\n")

    return prediction


def generate_full_sequence(model, image_path, task, max_steps=10):
    """
    Generate full test sequence.

    Args:
        model: Trained model
        image_path: Path to screenshot
        task: Task description
        max_steps: Maximum steps to generate

    Returns:
        List of generated steps
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    print("\n" + "="*60)
    print(f"GENERATING TEST SEQUENCE FOR: {task}")
    print("="*60)

    # Generate sequence
    sequence = model.generate_sequence(
        initial_image=image,
        question=task,
        max_steps=max_steps,
        max_length=128,
        num_beams=4
    )

    print("\n" + "="*60)
    print("GENERATED TEST SEQUENCE")
    print("="*60)

    for step in sequence:
        print(f"\nStep {step['step_num']}: {step['action']}")

    print("\n" + "="*60)
    print(f"Total steps: {len(sequence)}")
    print("="*60 + "\n")

    return sequence


def interactive_mode(model):
    """
    Interactive mode - keep predicting next steps.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Generate test steps interactively!")
    print("Type 'quit' to exit, 'reset' to start over")
    print("="*70 + "\n")

    history = []
    task = None
    image_path = None

    while True:
        # Get task (once)
        if task is None:
            task = input("Enter task description: ").strip()
            if task.lower() == 'quit':
                break

        # Get image (once)
        if image_path is None:
            image_path = input("Enter screenshot path: ").strip()
            if image_path.lower() == 'quit':
                break

            # Verify image exists
            if not Path(image_path).exists():
                print(f"❌ Image not found: {image_path}")
                image_path = None
                continue

        # Predict next step
        current = history[-1] if history else None
        try:
            prediction = predict_single_step(
                model=model,
                image_path=image_path,
                task=task,
                history=history,
                current_action=current
            )

            # Add to history
            history.append(prediction)

            # Ask if continue
            choice = input("\n[c]ontinue, [r]eset, or [q]uit? ").strip().lower()

            if choice == 'q' or choice == 'quit':
                break
            elif choice == 'r' or choice == 'reset':
                history = []
                task = None
                image_path = None
                print("\n✓ Reset! Starting new sequence...\n")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            choice = input("Try again? [y/n] ").strip().lower()
            if choice != 'y':
                break

    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="GUI Test Automation - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single step prediction
  python inference.py --image screenshot.png --task "Login to application"

  # With history
  python inference.py --image screenshot.png --task "Login" --history "Open homepage" "Click login button"

  # Generate full sequence
  python inference.py --image screenshot.png --task "Complete checkout" --sequence --max_steps 10

  # Interactive mode
  python inference.py --interactive
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='./outputs/best_model',
        help='Path to trained model (default: ./outputs/best_model)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to screenshot image'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Task description'
    )
    parser.add_argument(
        '--history',
        nargs='*',
        help='Previous actions (space-separated)'
    )
    parser.add_argument(
        '--current',
        type=str,
        help='Current action'
    )
    parser.add_argument(
        '--sequence',
        action='store_true',
        help='Generate full test sequence'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10,
        help='Maximum steps for sequence generation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )

    args = parser.parse_args()

    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)

    try:
        model = GUITestBLIP2(device=args.device)
        if Path(args.model).exists():
            model.load_model(args.model)
            print(f"✓ Loaded model from {args.model}")
        else:
            print(f"⚠ Model not found at {args.model}")
            print("Using base BLIP-2 model (not fine-tuned)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(model)
        return

    # Validate args
    if not args.image or not args.task:
        parser.print_help()
        print("\n❌ Error: --image and --task are required (unless using --interactive)")
        return

    # Check image exists
    if not Path(args.image).exists():
        print(f"\n❌ Error: Image not found: {args.image}")
        return

    # Generate
    if args.sequence:
        # Full sequence
        generate_full_sequence(
            model=model,
            image_path=args.image,
            task=args.task,
            max_steps=args.max_steps
        )
    else:
        # Single step
        predict_single_step(
            model=model,
            image_path=args.image,
            task=args.task,
            history=args.history,
            current_action=args.current
        )


if __name__ == "__main__":
    main()
