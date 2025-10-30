"""
Test script to verify dataset loading works correctly
"""

import sys
sys.path.append('./src')

from data.dataset import GUITestDataset


def test_history_formatting():
    """Test that history is formatted correctly."""

    print("\n" + "="*60)
    print("Testing Dataset History Formatting")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = GUITestDataset(split='train', max_history_length=10)

    # Test first few samples
    print("\nTesting first 3 samples:\n")

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        raw_data = sample['raw_data']

        print(f"Sample {i+1}:")
        print(f"  Workflow: {sample['workflow']}")
        print(f"  Raw history type: {type(raw_data['previousActionHistory'])}")
        print(f"  Raw history value: {repr(raw_data['previousActionHistory'][:100] if isinstance(raw_data['previousActionHistory'], str) else raw_data['previousActionHistory'])}")

        # Check input text
        input_text = sample['input_text']

        # Extract the history section
        if "Previous steps:" in input_text:
            start = input_text.index("Previous steps:") + len("Previous steps:")
            end = input_text.index("Current action:")
            history_section = input_text[start:end].strip()

            print(f"  Formatted history (first 200 chars):")
            print(f"    {repr(history_section[:200])}")

            # Check if it's not character-by-character
            if len(history_section) > 0:
                lines = [line for line in history_section.split('\n') if line.strip()]
                if lines:
                    first_line = lines[0]
                    # Should be "1. CLICK: ..." not "1. C"
                    if len(first_line) < 10:
                        print(f"  ⚠️ WARNING: First line too short: '{first_line}'")
                        print(f"     This might indicate character-by-character iteration!")
                    else:
                        print(f"  ✓ First line looks good: '{first_line[:50]}...'")

        print()

    print("="*60)
    print("Test complete!")
    print("="*60 + "\n")


def test_specific_sample():
    """Test the specific problematic sample."""

    print("\n" + "="*60)
    print("Testing Specific Sample")
    print("="*60)

    dataset = GUITestDataset(split='train')

    # Find Apollo workflow
    print("\nSearching for Apollo workflow samples...")
    apollo_samples = []

    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        if 'Apollo' in sample['workflow']:
            apollo_samples.append((i, sample))
            if len(apollo_samples) >= 3:
                break

    if apollo_samples:
        print(f"Found {len(apollo_samples)} Apollo samples\n")

        for idx, (i, sample) in enumerate(apollo_samples):
            print(f"Apollo Sample {idx+1} (index {i}):")
            print(f"  Task: {sample['raw_data']['question'][:80]}...")
            print(f"  Target: {sample['target_text'][:80]}...")

            # Check history formatting
            input_text = sample['input_text']
            if "Previous steps:" in input_text:
                start = input_text.index("Previous steps:")
                end = input_text.index("Current action:")
                history_section = input_text[start:end]

                lines = [line.strip() for line in history_section.split('\n') if line.strip() and line.strip() != 'Previous steps:']

                print(f"  History lines: {len(lines)}")
                if lines:
                    print(f"  First line: {lines[0]}")
                    if len(lines) > 1:
                        print(f"  Last line: {lines[-1]}")

                    # Validation
                    if any(len(line) < 5 for line in lines):
                        print(f"  ❌ PROBLEM: Some lines are too short (character-by-character issue)")
                    else:
                        print(f"  ✓ OK: All lines have reasonable length")

            print()
    else:
        print("No Apollo samples found in first 100 samples")

    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATASET TESTING SCRIPT")
    print("="*70)

    try:
        test_history_formatting()
        test_specific_sample()

        print("\n✓ All tests completed!")
        print("\nIf you see character-by-character issues (like '1. C' instead of '1. CLICK:'),")
        print("the dataset fix needs adjustment.\n")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed dependencies: pip install -r requirements.txt")
        print("2. Internet connection (to download GUIDE dataset)")
        print()
