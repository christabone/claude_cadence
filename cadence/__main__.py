"""
Main module entry point for Claude Cadence

Allows running as: python -m cadence
"""

import sys
from pathlib import Path

# Add parent directory to path to import orchestrate module
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrate import main

if __name__ == "__main__":
    main()
