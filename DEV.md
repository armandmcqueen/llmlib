# Development Guide for LLMLib

This guide explains how to set up your development environment and publish new versions of the LLMLib package.

## Setting Up Your Development Environment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llmlib.git
   cd llmlib
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the development dependencies:
   ```
   pip install flit pytest
   ```

4. Install the package in editable mode:
   ```
   flit install --symlink
   ```

## Running Tests

To run the tests:

```
pytest tests/
```

## Publishing the Package

To publish a new version of LLMLib:

1. Update the version number in `llmlib/__init__.py`:
   ```python
   __version__ = "x.y.z"  # Replace with the new version number
   ```

2. Commit your changes:
   ```
   git add llmlib/__init__.py
   git commit -m "Bump version to x.y.z"
   ```

3. Tag the new version:
   ```
   git tag vx.y.z
   git push origin main --tags
   ```

4. Build and publish the package:
   ```
   flit publish
   ```

   This command will build the package and upload it to PyPI. You'll need to have a PyPI account and have your credentials configured (see below).

## Configuring PyPI Credentials

To publish to PyPI, you need to set up your credentials. The recommended way is to use a PyPI API token:

1. Go to https://pypi.org and log in.
2. Go to your account settings and create a new API token.
3. Create a `~/.pypirc` file with the following content:
   ```
   [pypi]
   username = __token__
   password = your-api-token-here
   ```

Make sure to keep your API token secret and never commit it to version control.

## Dependencies

The main dependencies for LLMLib are:

- pydantic
- openai
- anthropic

These are listed in the `pyproject.toml` file and will be automatically installed when someone installs your package.

For development, you also need:

- flit (for building and publishing)
- pytest (for running tests)

These are not listed in `pyproject.toml` as they are not required for users of your library.

## Continuous Integration

Consider setting up a CI/CD pipeline (e.g., using GitHub Actions) to automatically run tests and potentially publish new versions when you push to the main branch or create a new tag.

Remember to always test your changes thoroughly before publishing a new version. It's also a good practice to maintain a changelog to keep track of changes between versions.