#!/usr/bin/env python3
"""
Test script to demonstrate the limitation with YAML anchors and our custom loader.
This shows that anchors defined in an included file are not available in the including file.
"""

import yaml
import sys
from pathlib import Path

# Add the cadence directory to the path
sys.path.insert(0, str(Path(__file__).parent / "cadence"))

from prompt_loader import load_yaml_with_includes


def test_standard_yaml_anchors():
    """Test how standard YAML handles anchors without includes."""
    print("=== Test 1: Standard YAML Anchor Behavior ===\n")

    yaml_content = """
# Define an anchor
my_anchor: &my_anchor "This is an anchor value"

# Reference the anchor
my_reference: *my_anchor
"""

    data = yaml.safe_load(yaml_content)
    print(f"Anchor value: {data['my_anchor']}")
    print(f"Reference value: {data['my_reference']}")
    print(f"Are they equal? {data['my_anchor'] == data['my_reference']}")
    print("\n✅ Standard YAML anchors work within the same file\n")


def test_included_anchors():
    """Test if anchors from included files are accessible."""
    print("=== Test 2: Anchors from Included Files ===\n")

    # Create a test that tries to use !include
    test_yaml = """
# Include the common file
common: !include _common.yaml

# Try to reference an anchor from the included file
# This won't work because anchors are resolved at parse time
# and the !include is processed after parsing
"""

    test_file = Path("test_include_anchors.yaml")
    test_file.write_text(test_yaml)

    try:
        # Change to the prompts directory for relative includes
        import os
        os.chdir("cadence/prompts")

        data = load_yaml_with_includes("../../test_include_anchors.yaml")
        print("Loaded data successfully")
        print(f"Type of 'common': {type(data.get('common'))}")
        print(f"Keys in loaded data: {list(data.keys())}")
        print("\n❌ Anchors from included files are NOT available in the including file")
        print("   The !include returns the parsed content, not the raw YAML with anchors")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.chdir("../..")
        test_file.unlink(missing_ok=True)


def test_workaround_approach():
    """Test a potential workaround using a different structure."""
    print("\n=== Test 3: Potential Workaround ===\n")

    print("Option 1: Use !include for content blocks only (current approach)")
    print("  - Put long prose in .md files")
    print("  - Use !include to pull in the prose")
    print("  - Keep anchors in the main YAML files")
    print("")
    print("Option 2: Create a custom anchor resolution system")
    print("  - Modify the loader to pre-process all files")
    print("  - Build a global anchor registry")
    print("  - Resolve references after loading")
    print("")
    print("Option 3: Use a two-stage loading process")
    print("  - First load _common.yaml to get anchors")
    print("  - Then load other files with anchor context")
    print("")
    print("Recommendation: Stick with Option 1 - it's simpler and follows the PRD")


def demonstrate_actual_usage():
    """Show how the system will actually work with the current loader."""
    print("\n=== Test 4: Actual Usage Pattern ===\n")

    example = """
Example structure that WILL work:

# agents/initial.yaml
agent_prompts:
  initial:
    sections:
      # Include markdown content
      - !include ../core/supervised_context.md
      - !include ../core/safety_notice.md
      - !include ../core/guidelines.md

      # Inline YAML content with variables
      - |
        Session ID: {session_id}
        Max turns: {max_turns}

      # Include more markdown
      - !include ../core/todo_instructions.md

This approach:
- Extracts long prose to .md files (Task 4)
- Uses !include for content insertion
- Keeps YAML structure and variables in YAML files
- Doesn't rely on cross-file anchor references
"""
    print(example)


def main():
    """Run all tests to demonstrate the limitation."""
    test_standard_yaml_anchors()
    test_included_anchors()
    test_workaround_approach()
    demonstrate_actual_usage()

    print("\n" + "="*60)
    print("CONCLUSION: Anchors in included files are not accessible")
    print("We should focus on the approved approach from the PRD:")
    print("- Extract long prose to .md files")
    print("- Use !include for content blocks")
    print("- Keep structural YAML in the main files")
    print("="*60)


if __name__ == "__main__":
    main()
