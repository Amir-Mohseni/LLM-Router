#!/bin/bash

# Simple script to run all pytest tests in the data_collection/tests directory
# Run this script from the project root directory.

echo "Running tests in data_collection/tests..."

# Use python -m pytest to ensure correct module discovery
# Add -v for verbose output (shows test names)
python -m pytest -v data_collection/tests

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "All tests passed!"
else
  echo "Some tests failed."
fi

exit $exit_code 