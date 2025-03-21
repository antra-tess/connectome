# Bot Framework Tests

This directory contains tests for the Bot Framework, organized by component layer.

## Running Tests

To run all tests:

```bash
cd bot_framework
python -m pytest tests/
```

To run tests for a specific layer:

```bash
python -m pytest tests/activity/
```

To run a specific test file:

```bash
python -m pytest tests/activity/test_client.py
```

To run a specific test:

```bash
python -m pytest tests/activity/test_client.py::TestSocketIOClient::test_init
```

## Coverage Report

To generate a coverage report:

```bash
pip install pytest-cov
python -m pytest --cov=bot_framework tests/
```

For HTML coverage report:

```bash
python -m pytest --cov=bot_framework --cov-report=html tests/
```

The HTML report will be generated in the `htmlcov` directory.

## Test Structure

The tests are organized by component layer:

- `activity/`: Tests for the activity layer (Socket.IO client and message handling)
- `environments/`: Tests for the environments layer
- `interface/`: Tests for the interface layer
- `utils/`: Tests for utility functions

## Writing Tests

When writing new tests, follow these guidelines:

1. Use pytest fixtures for common setup
2. Mock external dependencies
3. Keep test functions focused on testing a single aspect of functionality
4. Use descriptive test names
5. Include docstrings explaining what the test is verifying

Example:

```python
def test_something_specific(mock_dependency):
    """
    Test that specific functionality works as expected.
    
    This test verifies:
    1. Action X happens when condition Y is met
    2. The system handles edge case Z correctly
    """
    # Setup
    # ...
    
    # Execute
    # ...
    
    # Verify
    # ...
``` 