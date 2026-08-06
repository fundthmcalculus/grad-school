# Code Style Guidelines

This repository uses automated code style checking on pull requests to `main`.

## Current State

We are starting with **very lax controls** that check for:
- Basic formatting issues (trailing whitespace, blank line issues)
- Obvious syntax problems (flagged by Ruff)

Most checks are run in **informational mode** and do not block merges. This allows us to establish the practice without being overly restrictive.

## Tools

- **Black**: Python code formatter
  - Consistent, opinionated formatting
  - Configuration: `pyproject.toml`
  
- **Ruff**: Modern Python linter
  - Fast linting for error detection
  - Configuration: `pyproject.toml`

## Running Checks Locally

To check code style before pushing:

```bash
# Install tools
pip install black ruff

# Check formatting (without modifying)
black --check .

# Auto-fix formatting issues
black .

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
