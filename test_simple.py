"""
Simple standalone test for history formatting
No dependencies required
"""

import re


def format_history(history, max_history_length=10):
    """Format action history into readable text - standalone version."""

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
        history = history[-max_history_length:]

        # Format with numbering
        formatted = "\n".join([f"{i+1}. {action}" for i, action in enumerate(history)])
        return formatted

    # Fallback
    return "None"


def test():
    """Test the formatting function."""

    print("\n" + "="*70)
    print("HISTORY FORMATTING TEST")
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

    print("\n📥 INPUT (from dataset):")
    print("-" * 70)
    print(f"Type: {type(test_history).__name__}")
    print(f"Length: {len(test_history)} characters")
    print(f"First 100 chars: {test_history[:100]}...")
    print("-" * 70)

    # Format it
    formatted = format_history(test_history, max_history_length=10)

    print("\n📤 OUTPUT (formatted):")
    print("-" * 70)
    print(formatted)
    print("-" * 70)

    # Validate
    print("\n✅ VALIDATION:")
    lines = [line for line in formatted.split('\n') if line.strip()]

    print(f"  Number of actions: {len(lines)}")
    print(f"  First action: {lines[0][:60]}...")
    print(f"  Last action: {lines[-1][:60]}...")

    # Check for the bug
    avg_length = sum(len(line) for line in lines) / len(lines) if lines else 0

    print(f"\n  Average line length: {avg_length:.1f} characters")

    if avg_length < 10:
        print(f"  ❌ FAIL: Character-by-character bug detected!")
        return False
    else:
        print(f"  ✅ PASS: Lines have proper length")

    # Check content
    if all("CLICK:" in line or "TYPE:" in line for line in lines):
        print(f"  ✅ PASS: All actions contain keywords")
    else:
        print(f"  ❌ FAIL: Some actions missing keywords")
        return False

    # Check no character repetition
    if any(len(line) < 5 for line in lines):
        print(f"  ❌ FAIL: Some lines too short")
        return False
    else:
        print(f"  ✅ PASS: No short lines (no char-by-char iteration)")

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)

    print("\nBefore fix, you would see:")
    print("  1. t")
    print("  2. h")
    print("  3. e")
    print("  ...")
    print("\nNow you correctly see:")
    print("  1. CLICK: Click on the Search")
    print("  2. CLICK: Click on the More Filters")
    print("  ...")

    return True


if __name__ == "__main__":
    success = test()
    if success:
        print("\n✅ Fix is working correctly! 🎉\n")
    else:
        print("\n❌ Fix needs adjustment.\n")
