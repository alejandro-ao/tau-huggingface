"""Hugging Face-specific commands for Tau."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from urllib import parse, request

from tau_coding.extensions import ExtensionAPI, ExtensionCommandContext

_HUGGINGFACE_MODEL_API = "https://huggingface.co/api/models"
_ROUTE_RESET_ALIASES = {"automatic", "auto", "reset"}


def _current_route(api: ExtensionAPI) -> str:
    route = api.context.inference_provider or "automatic"
    return f"Hugging Face route: {route}"


def _fetch_available_routes(model: str) -> tuple[str, ...]:
    """Fetch live inference-provider routes advertised for a model."""
    if not model:
        return ()

    model_path = parse.quote(model, safe="/")
    url = f"{_HUGGINGFACE_MODEL_API}/{model_path}?expand%5B%5D=inferenceProviderMapping"
    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with request.urlopen(request.Request(url, headers=headers), timeout=10) as response:
        payload = json.load(response)

    mapping = payload.get("inferenceProviderMapping") if isinstance(payload, Mapping) else None
    if not isinstance(mapping, Mapping):
        return ()

    return tuple(
        sorted(
            provider
            for provider, details in mapping.items()
            if isinstance(provider, str)
            and isinstance(details, Mapping)
            and details.get("status") == "live"
        )
    )


async def _choose_route(api: ExtensionAPI) -> None:
    """Open the route picker and apply the selected provider."""
    try:
        try:
            routes = await asyncio.to_thread(_fetch_available_routes, api.context.model)
        except Exception as exc:  # noqa: BLE001 - provider metadata is optional
            api.notify(f"Could not load available Hugging Face providers: {exc}", "warning")
            routes = ()

        selected = await api.context.ui.select(
            "Hugging Face inference provider",
            ("automatic", *routes),
        )
        if selected is None:
            return

        selected_route = None if selected.casefold() in _ROUTE_RESET_ALIASES else selected
        message = api.set_inference_provider(selected_route)
        api.notify(f"Hugging Face route: {message}")
    except Exception as exc:  # noqa: BLE001 - background command tasks must not leak errors
        try:
            api.notify(f"Could not change Hugging Face route: {exc}", "error")
        except Exception:  # noqa: BLE001 - stale APIs cannot be notified
            return


def _route(args: str, context: ExtensionCommandContext) -> str | None:
    api = context.api
    if api.context.provider_name != "huggingface":
        return "/hf route requires the huggingface provider"

    value = args.strip()
    if not value:
        if not api.context.has_ui:
            return _current_route(api)
        try:
            asyncio.get_running_loop().create_task(_choose_route(api))
        except RuntimeError:
            return _current_route(api)
        return None

    selected_route = None if value.casefold() in _ROUTE_RESET_ALIASES else value
    selected = api.set_inference_provider(selected_route)
    return f"Hugging Face route: {selected}"


def _hf(args: str, context: ExtensionCommandContext) -> str | None:
    value = args.strip()
    if not value:
        return "Usage: /hf route [automatic|<inference-provider>]"

    command, _, command_args = value.partition(" ")
    if command.casefold() != "route":
        return f"Unknown /hf command: {command}. Available commands: route"
    return _route(command_args, context)


def setup(tau: ExtensionAPI) -> None:
    """Register Hugging Face-specific commands."""
    tau.register_command(
        "hf",
        _hf,
        description="Hugging Face commands.",
        usage="/hf route [automatic|<inference-provider>]",
    )
