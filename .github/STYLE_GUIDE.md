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
# Install linting tools
pip install black flake8 Flake8-pyproject mypy

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

We now use **casual enforcement** — checks enforce but with lenient rules:
- **Black**: Enforces consistent formatting (88-char line length)
- **Flake8**: Enforces basic linting with lenient configuration (120-char max, E203 ignored)
- **mypy**: Type checking enforced but lenient (ignores missing imports and type hints)

New code must pass these checks, but the lenient configuration means only the most obvious issues block merges. As the codebase stabilizes, rules can be tightened incrementally.

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
