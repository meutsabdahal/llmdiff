# Contributing to llmdiff

Thanks for your interest in contributing! This document covers how to set up a
development environment, the quality gates every change must pass, and the
conventions used in this repository.

## Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) running locally if you want to test the CLI
  against a real model (`ollama pull llama3.2` gives you a small default)

## Development workflow

Do not clone this repository directly; all contributions go through a fork.

1. **Fork the repository** using the Fork button on
   <https://github.com/meutsabdahal/llmdiff>.

2. **Clone your fork** and add the original repository as `upstream`:

   ```bash
   git clone https://github.com/<your-username>/llmdiff.git
   cd llmdiff
   git remote add upstream https://github.com/meutsabdahal/llmdiff.git
   ```

3. **Install dependencies:**

   ```bash
   uv sync --all-extras --group dev
   ```

   This installs the package in editable mode along with the optional
   semantic scoring dependencies and all development tools.

4. **Create a branch** from the latest upstream `dev`:

   ```bash
   git fetch upstream
   git checkout -b feature/my-change upstream/dev
   ```

   Use a short descriptive branch name such as `feature/seed-flag` or
   `fix/env-parsing`.

5. **Make your changes**, keeping each commit focused on one logical change.

6. **Run the quality gates** locally (see below) and make sure they pass.

7. **Push the branch to your fork:**

   ```bash
   git push -u origin feature/my-change
   ```

8. **Open a pull request against the `dev` branch** (not `main`). CI must be
   green before merging. Changes land in `main` later through a release PR
   from `dev`.

### Optional: local environment variables

Copy `.env.example` to `.env` if you need to set `HF_TOKEN` or the transformers
logging variables. Only the keys listed in `.env.example` are loaded. `.env` is
git-ignored; never commit tokens or other secrets.

## Quality gates

Please run these before pushing:

```bash
uv run pytest tests/ -v      # test suite
uv run ruff check .          # lint (includes unused imports and import order)
uv run mypy                  # type checking (no error-code overrides)
```

CI runs the same checks on every pull request, plus packaging checks that
maintainers take care of.

## Testing

- Add or update tests for every behavior change. Bug fixes should include a
  test that fails without the fix.
- Tests must not require a running Ollama instance or network access. Mock
  HTTP interactions instead (see `tests/test_runner.py` for the existing
  patterns).
- Async tests use `pytest-asyncio` with the `@pytest.mark.asyncio` marker.

To verify a change end-to-end against a real model:

```bash
uv run llmdiff --prompt-a a.txt --prompt-b b.txt --inputs cases.json \
  --model llama3.2 --temperature 0 --seed 42
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short imperative summary>

<optional body explaining what and why>
```

- Types in use: `feat`, `fix`, `refactor`, `perf`, `style`, `docs`, `test`,
  `chore`, `ci`.
- Scope is the affected module, e.g. `fix(runner): ...` or
  `feat(cli, config): ...`.
- Keep the summary under ~72 characters; use the body to explain the why.

## Code style

- Formatting and linting are enforced by ruff with the rules configured in
  `pyproject.toml`; run `uv run ruff check --fix .` to auto-fix imports.
- All code is type-annotated and must pass mypy without new overrides.
- User-facing CLI errors go through `typer.echo(..., err=True)` followed by
  `raise typer.Exit(1)`; expected failure modes must not surface as raw
  tracebacks.
- Files are read and written with explicit `encoding="utf-8"`.

## Reporting issues

- **Bugs / feature requests:** open an issue at
  <https://github.com/meutsabdahal/llmdiff/issues> with reproduction steps,
  expected vs. actual behavior, and your `llmdiff --version` output.
- **Security vulnerabilities:** please do not open a public issue. Report
  privately via [GitHub security advisories](https://github.com/meutsabdahal/llmdiff/security/advisories/new).

## Release process (maintainers)

1. Merge `dev` into `main` through a release PR.
2. Bump `version` in `pyproject.toml` and run `uv lock`.
3. Commit as `chore: release X.Y.Z`, push, and tag `vX.Y.Z`.
4. Build and verify: `uv build && uv run twine check --strict dist/*`.
5. Create the GitHub release with the artifacts attached and upload to PyPI.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
