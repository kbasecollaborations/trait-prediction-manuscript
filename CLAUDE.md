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
- Use `scripts/figure1/figure1c_plot.py` as a reference for plotting style and standards.
- Use a figure size of `figsize=(12, 6)` for a single subplot and `figsize=(12, 12)` for figures with multiple subplots unless there is a good reason to deviate.

## Machine learning guidelines

- Use a correlation filtering threshold of 0.95 and variance threshold of 1% (0.01) for feature selection.
- Use KOFAM features because it does slightly better than RAST (45 vs. 19)
- Use CatBoost (because it does better than RF most of the time) and make sure to use the `make_classifier` function from `scripts/ml.py` if possible, at least use the same parameters provided there.
- Disable `cat_features` as CatBoost handles 0/1 integers fine

## Manuscript editing (LaTeX): instructions & guardrails

- Put individual lines in the .tex files on separate lines for easier diffs
- Use academic writing style, avoid contractions, and ensure proper grammar and punctuation.

### Scope (what to edit)

- Primary goal: improve sentence structure, narrative flow, and conciseness **without changing technical meaning**.
- Edit prose only by default (Abstract/Intro/Results/Discussion), not figures/tables/math unless explicitly requested.
- Prefer “small, safe, local” edits over large rewrites unless asked.

### Hard constraints (must follow)

- Do NOT change: equations/math environments, numbers/results, variable names, symbols, units, theorem/lemma statements, labels (`\label{}`), refs (`\ref{}`, `\autoref{}`), citation keys (`\cite{}`), bibliography files, or cross-referencing structure.
- Do NOT add new claims, new references, or new citations. If something needs support, insert a `TODO(CITATION NEEDED)` comment only.
- Do NOT “smooth over” uncertainty: preserve hedging level (e.g., “may”, “suggests”, “consistent with”) and avoid strengthening causal language.
- Do NOT edit custom macros or packages unless requested; treat them as API.

### Workflow (required)

1. First, briefly list the 3–7 biggest issues you see in the provided text (redundancy, unclear antecedents, paragraph topic drift, etc.).
2. Then propose a plan: what you will change, what you will not change, and what sections/files you will touch.
3. Then apply edits.
4. After edits, run a self-check:
   - “Meaning preserved?” (yes/no + note any risky sentences)
   - “Constraints violated?” (yes/no)
   - “Any ambiguous terms introduced?” (list)

### Style targets (soft constraints)

- Prefer active voice when it improves clarity, but keep passive voice when agent/action is unknown or irrelevant.
- Reduce nominalizations and long prepositional chains.
- Keep terminology consistent (do not introduce synonyms for key technical terms).
- Prefer shorter sentences, but don’t split if it harms logic.

### Output format

- If editing files: make changes directly and keep diffs minimal.
- If not editing files: return a unified diff or “before/after” blocks for each paragraph you changed.
- For any non-trivial rewrite, include a 1-line rationale.

## General behavior

- Prefer readability and maintainability over clever one-liners.
- When modifying existing code, keep the existing style unless this file says otherwise.
- Do not read the complete file contents of anything in data/ folder. Most of those files are large and reading them fully is inefficient. Instead read only the first couple of lines and if the number of columns is large, then read only the first couple of columns.

## Script run instructions

- Use `uv run python -m scripts.<module>.<script>` to run scripts.
