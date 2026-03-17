# Repository Guidelines

## Project Structure & Module Organization
This repository is notebook-first. Authoritative content lives under `docs/`, which is rendered as a Quarto website.

- `docs/00_Lab_Guide/`: repo map and working protocol
- `docs/01_Core_Path/`: primary learning sequence
- `docs/10_Research_Tracks/`: optional advanced branches
- `docs/20_Frontier_Notes/`: literature-first exploratory notes
- `docs/quarto/` and `docs/javascripts/`: shared site styling and browser-side helpers
- `_quarto.yml`: site navigation, render targets, and output settings
- `site-quarto/`: generated HTML output; do not hand-edit

Keep new notebooks in the lane that matches their role, and preserve the numbered filename pattern such as `03_boundary_and_initial_conditions.ipynb`.

## Build, Test, and Development Commands
- `pip install -e .`: install runtime dependencies for notebooks and JupyterLab
- `pip install -e ".[dev]"`: install developer tools, including `pytest`, `ruff`, and `quarto-cli`
- `jupyter lab`: open notebooks for authoring and execution
- `quarto preview`: start the local Quarto site with live reload
- `quarto render --to html`: build the static site into `site-quarto/`
- `pytest`: run automated tests when helper modules or scripts are added
- `ruff check .`: lint Python code and notebook-adjacent scripts

## Coding Style & Naming Conventions
Use Python 3.10+ with 4-space indentation and standard PEP 8 naming: `snake_case` for functions, variables, and notebook filenames. Keep notebooks narrow: one topic, one canonical problem, one clear experiment thread. Put reusable styling in `docs/quarto/styles.scss` and shared HTML hooks in `docs/quarto/includes.html` instead of duplicating them across notebooks.

## Testing Guidelines
There is no large committed test suite yet. For any new Python utilities, add tests under `tests/` using `test_*.py` naming and run `pytest` locally before opening a PR. For notebook changes, verify the affected notebook renders cleanly through Quarto; use `quarto render path/to/notebook.ipynb --execute` when outputs must be regenerated.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative commit subjects, usually with prefixes like `feat:` and `fix:`. Follow that pattern when possible, for example `feat: add adaptive sampling notebook scaffold`.

PRs should summarize scope, list touched notebooks or site assets, link any issue, and include screenshots when changing homepage or Quarto styling. Call out whether the change affects rendered output, navigation, or deployment behavior.
