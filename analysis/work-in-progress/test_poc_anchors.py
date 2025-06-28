#!/usr/bin/env python3
"""
POC Test Script - Validates YAML anchor handling with custom loader
Tests whether anchors defined in _common.yaml can be properly referenced
when included in other files using the custom YAML loader from Task 2.
"""

import os
import sys
import json
from pathlib import Path

# Add the cadence directory to the path to import our custom loader
sys.path.insert(0, str(Path(__file__).parent / "cadence"))

# Import the custom YAML loader from Task 2
from prompt_loader import load_yaml_with_includes


def test_anchor_handling():
    """Test whether the custom loader properly handles YAML anchors in included files."""

    print("=== POC YAML Anchor Validation Test ===\n")

    # Path to our test file
    test_file = Path(__file__).parent / "cadence" / "prompts" / "poc_test.yaml"

    try:
        # Load the YAML file with includes
        print(f"Loading test file: {test_file}")
        data = load_yaml_with_includes(str(test_file))

        print("\n✅ Successfully loaded YAML with includes!\n")

        # Test 1: Simple alias
        print("Test 1 - Simple alias reference:")
        print(f"  Expected: 'This is a simple test anchor'")
        print(f"  Actual: '{data.get('test_simple_alias')}'")
        assert data.get('test_simple_alias') == "This is a simple test anchor"
        print("  ✅ PASSED\n")

        # Test 2: Base paths
        print("Test 2 - Base paths reference:")
        paths = data.get('paths_test', {})
        print(f"  cadence_dir: {paths.get('cadence_dir')}")
        print(f"  scratchpad_dir: {paths.get('scratchpad_dir')}")
        assert paths.get('cadence_dir') == ".cadence"
        assert "{project_path}" in paths.get('scratchpad_dir', '')
        print("  ✅ PASSED\n")

        # Test 3: Nested anchor
        print("Test 3 - Nested anchor reference:")
        nested = data.get('nested_test', {})
        print(f"  Has base_path: {'base_path' in nested}")
        print(f"  Message: {nested.get('message')}")
        assert 'base_path' in nested
        assert nested.get('message') == "Testing nested anchor reference"
        print("  ✅ PASSED\n")

        # Test 4: Merge key
        print("Test 4 - Merge key usage:")
        merge = data.get('merge_test', {})
        print(f"  Has project_path: {'project_path' in merge}")
        print(f"  Has custom_field: {'custom_field' in merge}")
        print(f"  Custom field value: {merge.get('custom_field')}")
        assert 'project_path' in merge  # From merged anchor
        assert merge.get('custom_field') == "additional value"
        print("  ✅ PASSED\n")

        # Test 5: Complex structures
        print("Test 5 - Complex structure with anchors:")
        agent = data.get('agent_config', {})
        print(f"  Has activation: {'activation' in agent}")
        print(f"  Has safety: {'safety' in agent}")
        print(f"  Activation starts with: {agent.get('activation', '')[:40]}...")
        assert "SERENA MCP ACTIVATION" in agent.get('activation', '')
        assert "IMPORTANT SAFETY NOTICE" in agent.get('safety', '')
        print("  ✅ PASSED\n")

        # Test 6: Template usage
        print("Test 6 - Template usage:")
        scratchpad = data.get('scratchpad_init', {})
        template = scratchpad.get('template', '')
        print(f"  Template contains 'Task Execution Scratchpad': {'Task Execution Scratchpad' in template}")
        print(f"  Completion protocol present: {'completion' in scratchpad}")
        assert 'Task Execution Scratchpad' in template
        assert 'COMPLETION PROTOCOL' in scratchpad.get('completion', '')
        print("  ✅ PASSED\n")

        # Test 7: Multiple anchor combination
        print("Test 7 - Combining multiple anchors:")
        full = data.get('full_context', {})
        print(f"  Keys present: {list(full.keys())}")
        assert 'project_path' in full  # From var_project_context
        assert 'completed_count' in full  # From var_status_tracking
        assert 'paths' in full  # From base_paths
        print("  ✅ PASSED\n")

        # Test 8: Code review templates
        print("Test 8 - Code review template references:")
        reviews = data.get('reviews', {})
        task_review = reviews.get('task_review', '')
        print(f"  Task review contains 'CODE REVIEW INSTRUCTIONS': {'CODE REVIEW INSTRUCTIONS' in task_review}")
        print(f"  Contains o3 model reference: {'model=\"o3\"' in task_review}")
        assert 'CODE REVIEW INSTRUCTIONS' in task_review
        assert 'model="o3"' in task_review
        print("  ✅ PASSED\n")

        # Test 9: Status templates
        print("Test 9 - Status template references:")
        status = data.get('status_messages', {})
        print(f"  Complete: {status.get('complete')}")
        print(f"  Progress: {status.get('progress')}")
        assert status.get('complete') == "Status: COMPLETE ✅"
        assert status.get('progress') == "Status: IN_PROGRESS"
        print("  ✅ PASSED\n")

        # Test 10: Complex nested with multiple merges
        print("Test 10 - Complex nested structure:")
        supervisor = data.get('supervisor_config', {})
        context = supervisor.get('context', {})
        print(f"  Context has project_path: {'project_path' in context}")
        print(f"  Context has completed_count: {'completed_count' in context}")
        print(f"  Has paths reference: {'paths' in supervisor}")
        assert 'project_path' in context
        assert 'completed_count' in context
        assert supervisor.get('paths', {}).get('cadence_dir') == '.cadence'
        print("  ✅ PASSED\n")

        print("="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe custom YAML loader successfully handles:")
        print("- Simple alias references (*anchor_name)")
        print("- Nested anchor references")
        print("- Merge key syntax (<<: *anchor_name)")
        print("- Multiple anchor combinations")
        print("- Complex nested structures")
        print("- All anchor types from _common.yaml")
        print("\n✅ POC VALIDATION SUCCESSFUL!")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the POC validation test."""
    success = test_anchor_handling()

    if success:
        print("\n📝 Next Steps:")
        print("1. The custom loader handles anchors correctly - no modifications needed")
        print("2. Can proceed with full _common.yaml implementation")
        print("3. Anchors work seamlessly with !include directives")
    else:
        print("\n⚠️  Issues found - may need loader modifications")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
