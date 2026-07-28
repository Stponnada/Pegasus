# Repository Guidelines

## Project Structure & Module Organization

This repository combines three related areas:

- `Documentation/` contains the design record. Treat `ontology_memory_architecture.md` (v0.2.1) as the authoritative specification; the PDFs and notebook provide background.
- `engine/src/ontomem/` contains the standalone Python 3.11+ memory engine. Tests live in `engine/tests/`, demos in `engine/demo/`, and the OpenCode adapter notes in `engine/adapters/opencode/`.
- `opencode/` is a separately versioned Bun/TypeScript host checkout. Follow its own `AGENTS.md` and package-level guides when editing it.

Keep host-specific integration out of the Python engine. The engine exposes a plain-data HTTP contract through `ontomem.service`.

## Build, Test, and Development Commands

Run engine commands from `engine/`:

```bash
uv venv && uv pip install -e ".[dev,llm]"  # create the environment and install extras
uv run pytest                              # run the hermetic test suite
uv run pytest tests/test_store.py          # run one focused test module
uv run pytest -m integration               # run live Gemini tests
uv run python -m ontomem.service           # start the HTTP service on port 8765
PYTHONPATH=src uv run python demo/server.py # launch the local read-pipeline demo
```

For host development, run `bun install` and `bun dev` from `opencode/`. Never run its root `bun test`; run tests and `bun typecheck` from the affected package directory.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and small, single-purpose modules for Python. Name modules, functions, and variables `snake_case`; classes and dataclasses use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Preserve architectural boundaries among extraction, merge, retrieval, persistence, and service layers. TypeScript changes must follow `opencode/AGENTS.md`; formatting there uses Prettier with no semicolons and a 120-column width.

## Testing Guidelines

Pytest discovers `engine/tests/test_*.py`. Add focused tests with every behavior change and keep deterministic logic hermetic. Mark network-dependent Gemini coverage with `@pytest.mark.integration`; it requires `GEMINI_API_KEY` and is excluded by default.

## Commit & Pull Request Guidelines

The parent repository has no commit history yet, so no root convention is established. Use short, imperative subjects (for example, `Add graph snapshot validation`). Inside `opencode/`, follow its Conventional Commit style, such as `fix(plugin): handle service timeout`.

Pull requests should explain scope and architectural impact, link relevant issues or spec sections, list commands run, and include screenshots for viewer or UI changes. Never commit API keys or populated `.env` files.
