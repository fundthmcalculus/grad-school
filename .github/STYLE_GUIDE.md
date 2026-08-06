# Code Style Guidelines

This repository uses automated code style checking on pull requests to `main`.

## Current State

We are starting with **very lax controls** that check for:
- Basic formatting issues (trailing whitespace, blank line issues)
- Obvious syntax problems (flagged by Ruff)

Most checks are run in **informational mode** and do not block merges. This allows us to establish the practice without being overly restrictive.

## Tools

- **Ruff**: Modern Python linter and formatter
  - Fast, zero-config setup
  - Provides automatic formatting suggestions
  - Configuration: `pyproject.toml`

## Running Checks Locally

To check code style before pushing:

```bash
# Install ruff
pip install ruff

# Check formatting (without modifying)
ruff format --check .

# Auto-fix formatting issues
ruff format .

# Run linter checks
ruff check .
```

## Future Tightening

As the codebase stabilizes, we will gradually introduce stricter checks:
1. Import organization
2. Naming conventions
3. Code complexity limits
4. Type hinting standards
5. Docstring requirements

This is similar to how we managed style in `tribble-fis`, `optimizers`, and `clustering` repositories.

## Configuration

Style settings are configured in `pyproject.toml` under the `[tool.ruff]` section. Changes to the enforcement level require discussion with the team.
