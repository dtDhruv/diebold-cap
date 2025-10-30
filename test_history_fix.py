"""
Quick test for the history formatting fix
Tests with the exact format from the dataset
"""

import sys
sys.path.append('./src')

from data.dataset import GUITestDataset


def test_multiline_string():
    """Test with exact format from dataset."""

    print("\n" + "="*70)
    print("Testing History Formatting Fix")
    print("="*70)

    # Your exact example from the dataset
    test_history = """1. CLICK: Click on the Search
2. CLICK: Click on the More Filters
3. TYPE: Type Job Titles in the Search filters
4. CLICK: Click on the Job Titles
5. CLICK: Click on the Search for a job title
6. TYPE: Type product managers in the Search for a job title
7. CLICK: Click on the Search filters
8. TYPE: Type total years of experience in the Search filters
9. CLICK: Click on the Total Years of Experience
10. CLICK: Click on the Min year"""

    print("\nInput (multi-line string):")
    print("-" * 70)
    print(f"Type: {type(test_history)}")
    print(f"Length: {len(test_history)} characters")
    print(f"Lines: {len(test_history.split(chr(10)))} lines")
    print("-" * 70)

    # Create a dummy dataset instance to access the method
    dataset = GUITestDataset(split='train', max_history_length=10)

    # Test the formatting
    formatted = dataset._format_history(test_history)

    print("\nOutput (formatted):")
    print("-" * 70)
    print(formatted)
    print("-" * 70)

    # Validation
    print("\nValidation:")
    lines = [line for line in formatted.split('\n') if line.strip()]

    print(f"  ✓ Number of actions: {len(lines)}")
    print(f"  ✓ Max history length: {dataset.max_history_length}")
    print(f"  ✓ Actions kept: {min(len(lines), dataset.max_history_length)}")

    if lines:
        print(f"\n  First action: {lines[0]}")
        print(f"  Last action: {lines[-1]}")

        # Check for the character-by-character bug
        avg_length = sum(len(line) for line in lines) / len(lines)
        print(f"\n  Average line length: {avg_length:.1f} characters")

        if avg_length < 10:
            print(f"  ❌ PROBLEM: Lines too short (character-by-character issue!)")
        else:
            print(f"  ✅ OK: Lines have reasonable length (proper actions)")

        # Check first action content
        if "CLICK:" in lines[0] or "TYPE:" in lines[0]:
            print(f"  ✅ OK: Actions contain proper keywords")
        else:
            print(f"  ❌ PROBLEM: Actions don't contain expected keywords")

    print("\n" + "="*70)


def test_with_limit():
    """Test that max_history_length works."""

    print("\n" + "="*70)
    print("Testing History Length Limit")
    print("="*70)

    test_history = """1. CLICK: Action 1
2. CLICK: Action 2
3. CLICK: Action 3
4. CLICK: Action 4
5. CLICK: Action 5
6. CLICK: Action 6
7. CLICK: Action 7
8. CLICK: Action 8
9. CLICK: Action 9
10. CLICK: Action 10
11. CLICK: Action 11
12. CLICK: Action 12"""

    # Test with max_history_length=10
    dataset = GUITestDataset(split='train', max_history_length=10)
    formatted = dataset._format_history(test_history)

    lines = [line for line in formatted.split('\n') if line.strip()]

    print(f"\nInput: 12 actions")
    print(f"Max history: {dataset.max_history_length}")
    print(f"Output lines: {len(lines)}")

    if len(lines) == dataset.max_history_length:
        print(f"✅ OK: Correctly limited to {dataset.max_history_length} actions")
        print(f"\nFirst action in output: {lines[0]}")
        print(f"Last action in output: {lines[-1]}")

        # Should keep the LAST 10 actions (3-12, not 1-10)
        if "Action 3" in lines[0]:
            print(f"✅ OK: Kept most recent actions (3-12)")
        elif "Action 1" in lines[0]:
            print(f"⚠️  Note: Kept oldest actions (1-10) instead of most recent")
    else:
        print(f"❌ PROBLEM: Expected {dataset.max_history_length} lines, got {len(lines)}")

    print("\n" + "="*70)


def test_edge_cases():
    """Test edge cases."""

    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)

    dataset = GUITestDataset(split='train', max_history_length=10)

    test_cases = [
        ("None", "None value"),
        ("", "Empty string"),
        (None, "None type"),
        ([], "Empty list"),
        ("Single action without numbering", "Single action"),
    ]

    for test_input, description in test_cases:
        result = dataset._format_history(test_input)
        print(f"\n  {description}:")
        print(f"    Input: {repr(test_input)}")
        print(f"    Output: {repr(result)}")

        if result in ["None", "Single action without numbering"]:
            print(f"    ✅ OK")
        else:
            print(f"    ⚠️  Unexpected output")

    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HISTORY FORMATTING FIX - VERIFICATION")
    print("="*70)

    try:
        # Run tests
        test_multiline_string()
        test_with_limit()
        test_edge_cases()

        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)

        print("\nSummary:")
        print("  - Multi-line string parsing: Tested")
        print("  - History length limit: Tested")
        print("  - Edge cases: Tested")
        print("\nIf all checks show ✅, the fix is working correctly!")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
