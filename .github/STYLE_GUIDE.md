# Code Style Guidelines

This repository uses automated code style checking on pull requests to `main`, matching the approach in `optimizers` and `clustering` repositories.

## Tools

- **Black**: Python code formatter
  - Consistent, opinionated formatting (88-char default line length)
  - Configuration: `pyproject.toml` under `[tool.black]`
  
- **Flake8**: Python linter (via Flake8-pyproject plugin)
  - Detects logical errors and style violations
  - Configuration: `pyproject.toml` under `[tool.flake8]`
  - Line length set to 120 to catch only lines Black cannot split

- **mypy**: Static type checker
  - Lenient baseline for now (gradual typing approach)
  - Will be ratcheted up to stricter checks over time
  - Configuration: `pyproject.toml` under `[tool.mypy]`

## Running Checks Locally

To check code style before pushing:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Check formatting (without modifying)
black --check .

# Auto-fix formatting issues
black .

# Run linter checks
flake8 .

# Run type checker
mypy .
```

## Current Enforcement Level

We start with **lax controls**:
- **Black**: Formats consistently but doesn't block merges
- **Flake8**: Checks for obvious errors and style violations (lenient configuration)
- **mypy**: Type checking is informational; ignores missing type hints and imports

## Future Tightening

As the codebase stabilizes, we will gradually tighten enforcement:
1. Stricter mypy settings (ratcheted one module at a time)
2. Increased flake8 complexity limits
3. Type hint requirements
4. Import organization standards

Changes to the enforcement level will be discussed with the team and updated in this guide.

## Configuration

All settings are in `pyproject.toml`:
- `[tool.black]` - formatter settings
- `[tool.flake8]` - linter settings  
- `[tool.mypy]` - type checker settings
