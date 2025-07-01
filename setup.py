#!/usr/bin/env python3
"""
Setup script for Claude Cadence
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="claude-cadence",
    version="0.1.0",
    author="Christopher Tabone",
    description="Checkpoint-based supervision framework for Claude Code agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christabone/claude_cadence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core functionality uses only standard library
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pyfakefs>=5.2.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "rich": [
            "rich>=13.0.0",  # For beautiful terminal output
        ],
    },
    entry_points={
        "console_scripts": [
            "cadence=cadence.__main__:main",
        ],
    },
    include_package_data=True,
)
