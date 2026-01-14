#!/usr/bin/env python3
"""Run all tests."""

import subprocess
import sys
from pathlib import Path

def main():
    test_dir = Path(__file__).parent / "tests"
    test_files = list(test_dir.glob("test_*.py"))

    if not test_files:
        print("No test files found")
        return 1

    failed = 0
    for test_file in test_files:
        print(f"\n{'='*50}")
        print(f"Running {test_file.name}")
        print('='*50)
        result = subprocess.run([sys.executable, str(test_file)], cwd=Path(__file__).parent)
        if result.returncode != 0:
            failed += 1

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {failed} test file(s) had failures")
    else:
        print("All tests passed!")
    return failed


if __name__ == "__main__":
    sys.exit(main())
