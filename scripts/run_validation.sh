#!/bin/bash

# Script to run ONLY the slow, full dataset validation tests.
# Run this script from the project root directory.

echo "Running FULL dataset validation tests (marked as slow)..."

# Use python -m pytest to ensure correct module discovery
# Add -v for verbose output
# Use -m slow to run only tests marked as slow
python -m pytest -v -m slow data_collection/tests/test_full_dataset_validation.py

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "Full dataset validation passed!"
else
  echo "Full dataset validation FAILED."
fi

exit $exit_code 