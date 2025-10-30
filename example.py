"""
Simple example script demonstrating GUI Test Automation

This shows how to use the trained model for predictions.
"""

import sys
sys.path.append('./src')

from models.blip2_model import GUITestBLIP2
from PIL import Image
import torch


def example_single_prediction():
    """Example: Predict single next action."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Step Prediction")
    print("="*70)

    # Load model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GUITestBLIP2(device=device)

    # For demo, we'll use the base model
    # In practice, load your fine-tuned model:
    # model.load_model('./outputs/best_model')

    print(f"Model loaded on {device}\n")

    # Create example scenario
    # In practice, load real screenshot: image = Image.open('screenshot.png')
    print("Creating example scenario...")
    print("Task: Login to application")
    print("Current screen: Login page")
    print("Previous steps: None (starting workflow)\n")

    # For demo, create a dummy image
    # Replace with actual screenshot in practice
    dummy_image = Image.new('RGB', (800, 600), color=(240, 240, 240))

    prompt = """Task: Login to the application

Previous steps:
None

Current action: None

Predict the next action:"""

    print("Generating prediction...")
    prediction = model.generate(
        images=[dummy_image],
        prompts=[prompt],
        max_length=50,
        num_beams=4
    )[0]

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Predicted next action: {prediction}")
    print("="*70 + "\n")


def example_sequence_generation():
    """Example: Generate full test sequence."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Sequence Generation")
    print("="*70)

    # Load model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GUITestBLIP2(device=device)

    print(f"Model loaded on {device}\n")

    # Create example scenario
    print("Creating example scenario...")
    print("Task: Complete user registration")
    print("Current screen: Registration form\n")

    # Dummy image for demo
    dummy_image = Image.new('RGB', (800, 600), color=(240, 240, 240))

    print("Generating test sequence...")
    sequence = model.generate_sequence(
        initial_image=dummy_image,
        question="Complete user registration",
        max_steps=5,
        max_length=50,
        num_beams=4
    )

    print("\n" + "="*70)
    print("GENERATED TEST SEQUENCE")
    print("="*70)

    for step in sequence:
        print(f"\nStep {step['step_num']}: {step['action']}")

    print("\n" + "="*70)
    print(f"Total steps generated: {len(sequence)}")
    print("="*70 + "\n")


def example_with_history():
    """Example: Prediction with action history."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Prediction with History")
    print("="*70)

    # Load model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GUITestBLIP2(device=device)

    print(f"Model loaded on {device}\n")

    # Create scenario with history
    print("Creating example scenario...")
    print("Task: Search for a product")
    print("Previous actions performed:")
    history = [
        "Open homepage",
        "Click on search icon",
        "Type 'laptop' in search box"
    ]

    for i, action in enumerate(history, 1):
        print(f"  {i}. {action}")

    print()

    # Dummy image
    dummy_image = Image.new('RGB', (800, 600), color=(240, 240, 240))

    # Format history
    history_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])

    prompt = f"""Task: Search for a product

Previous steps:
{history_text}

Current action: Type 'laptop' in search box

Predict the next action:"""

    print("Generating prediction...")
    prediction = model.generate(
        images=[dummy_image],
        prompts=[prompt],
        max_length=50,
        num_beams=4
    )[0]

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Predicted next action: {prediction}")
    print("="*70 + "\n")


def show_usage():
    """Show how to use with real screenshots."""
    print("\n" + "="*70)
    print("USAGE WITH REAL SCREENSHOTS")
    print("="*70)

    print("""
To use with your own screenshots:

1. Train the model first:
   python train.py

2. Use the inference script:
   python inference.py --image screenshot.png --task "Login to app"

3. Or in Python:
   from src.models.blip2_model import GUITestBLIP2
   from PIL import Image

   # Load model
   model = GUITestBLIP2(device='cuda')
   model.load_model('./outputs/best_model')

   # Load your screenshot
   image = Image.open('your_screenshot.png')

   # Create prompt
   prompt = '''Task: Your task here
   Previous steps:
   1. Previous action
   Current action: Last action
   Predict the next action:'''

   # Generate
   prediction = model.generate([image], [prompt])
   print(prediction[0])

For more examples, see:
- notebooks/GUI_Test_Automation_Training.ipynb
- QUICKSTART.md
- README.md
""")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GUI TEST AUTOMATION - EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates how to use the trained model.")
    print("Note: Using base BLIP-2 for demo. Train first for real results!")
    print("\nExamples:")
    print("1. Single step prediction")
    print("2. Full sequence generation")
    print("3. Prediction with history")
    print("="*70)

    try:
        # Run examples
        example_single_prediction()
        example_sequence_generation()
        example_with_history()
        show_usage()

        print("\n✓ All examples completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed dependencies: pip install -r requirements.txt")
        print("2. GPU available (or set device='cpu')")
        print("3. Internet connection (to download base model)")
        print("\nFor help, see README.md or QUICKSTART.md\n")


if __name__ == "__main__":
    main()
