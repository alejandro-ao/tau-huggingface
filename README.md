# Hugging Face extension for Tau

Hugging Face-specific commands for [Tau](https://github.com/huggingface/tau).

## Compatibility

Requires a Tau build containing `ExtensionAPI.set_inference_provider` and
`ExtensionContext.inference_provider`. Until that API is released, install Tau
from `main` or the pull request that introduces it.

## Install and use

```bash
git clone https://github.com/alejandro-ao/tau-huggingface.git
cd tau-huggingface
tau -e .
```

You can keep using Tau from another working directory by passing the absolute
clone path:

```bash
tau -e ~/repos/tau-huggingface
```

Commands:

- `/hf route` — open a picker with the live Hugging Face providers available
  for the active model.
- `/hf route <provider>` — pin the active session to a route.
- `/hf route automatic` — reset automatic routing; aliases: `auto`, `reset`.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```
