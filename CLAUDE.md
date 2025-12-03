# Claude instructions

## Python style (IMPORTANT)

- YOU MUST always write Python code with explicit type hints on all function signatures, methods, and class attributes where practical.
- YOU MUST target modern Python type hints (python 3.13+), use `typing` and `collections.abc` where appropriate.
- YOU MUST write docstrings for all non-trivial functions, classes, and methods in NumPy-style format.
- In NumPy-style docstrings, include at least the following sections when applicable:
  - Parameters
  - Returns or Yields
  - Raises
- Prefer clear, explicit types and parameter descriptions over brevity.

## Visualization settings (IMPORTANT)

- We are making figures for journal articles so aesthetics matter. Make sure to use good color schemes, fonts, and layouts.
- Use `scienceplots` as the default plotting style for matplotlib.
- When creating plots, always include axis labels, titles, and legends where appropriate.
- Ensure plots are properly sized for readability (e.g., larger fonts, appropriate figure size).

## General behavior

- Prefer readability and maintainability over clever one-liners.
- When modifying existing code, keep the existing style unless this file says otherwise.
