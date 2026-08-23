# Hugging Face extension for Tau

Hugging Face-specific commands for [Tau](https://github.com/huggingface/tau).

## Compatibility

Requires a Tau build containing `ExtensionAPI.set_inference_provider` and
`ExtensionContext.inference_provider`. The provider-status section additionally
uses the sidebar API merged in
[huggingface/tau#639](https://github.com/huggingface/tau/pull/639). Until that API
is released, install Tau from `main`. On older or non-interactive hosts, the
sidebar contribution is a safe no-op and `/hf route` remains available.

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

Hugging Face sessions show a compact sidebar section with the active model,
automatic or fixed route, selected provider, and live providers advertised by
the model API. It shows loading, unavailable, and no-live-provider states,
refreshes on model and route changes, and disappears when another Tau provider
is active.

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
uv run mypy extension.py tests
```
