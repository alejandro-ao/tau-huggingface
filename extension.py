"""Hugging Face-specific commands and session status for Tau."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib import parse, request

from rich.markup import escape
from tau_coding.extensions import ExtensionAPI, ExtensionCommandContext, ExtensionContext

_HUGGINGFACE_MODEL_API = "https://huggingface.co/api/models"
_ROUTE_RESET_ALIASES = {"automatic", "auto", "reset"}
_METADATA_TTL_SECONDS = 60.0
_STATE_POLL_SECONDS = 0.5
_MAX_VISIBLE_ROUTES = 5
_SIDEBAR_KEY = "provider-status"


@dataclass(frozen=True, slots=True)
class _ProviderMetadata:
    model: str
    routes: tuple[str, ...]
    error: str | None
    fetched_at: float


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


class _HuggingFaceExtension:
    def __init__(self, api: ExtensionAPI) -> None:
        self.api = api
        self._metadata: _ProviderMetadata | None = None
        self._metadata_task: asyncio.Task[_ProviderMetadata] | None = None
        self._metadata_model: str | None = None
        self._sidebar_task: asyncio.Task[None] | None = None
        self._observer_task: asyncio.Task[None] | None = None
        self._observed_state: tuple[str, str, str, str | None] | None = None
        self._automatic_response_route: str | None = None
        self._sidebar_render: tuple[str, tuple[str, ...]] | None = None

    def register(self) -> None:
        self.api.register_command(
            "hf",
            self._hf,
            description="Hugging Face commands.",
            usage="/hf route [automatic|<inference-provider>]",
        )
        self.api.on("session_start", self._on_session_start)
        self.api.on("session_shutdown", self._on_session_shutdown)
        for event_name in ("agent_start", "message_end", "agent_end"):
            self.api.on(event_name, self._on_session_state_event)

    def _sidebar(self, context: ExtensionContext | None = None) -> object | None:
        active_context = context or self.api.context
        ui = active_context.ui
        return getattr(ui, "sidebar", None)

    def _state(self) -> tuple[str, str, str, str | None]:
        context = self.api.context
        return (
            context.provider_name,
            context.model,
            context.inference_provider_mode,
            context.inference_provider,
        )

    def _content(self, metadata: _ProviderMetadata | None) -> list[str]:
        context = self.api.context
        selected = context.inference_provider
        if selected is None and context.inference_provider_mode == "automatic":
            selected = self._automatic_response_route
        if selected is None:
            route_status = "[dim]○ automatic routing[/dim]"
        else:
            route_status = (
                f"[green]●[/green] {context.inference_provider_mode} via {escape(selected)}"
            )
        lines = [f"[b]{escape(context.model)}[/b]", route_status]

        if metadata is None:
            return [*lines, "[dim]Loading providers…[/dim]"]
        if metadata.error is not None:
            return [*lines, "[yellow]Providers unavailable[/yellow]"]
        if not metadata.routes:
            return [*lines, "[dim]No live providers[/dim]"]

        lines.append("[dim]available providers[/dim]")
        routes = list(metadata.routes)
        if selected in routes:
            routes.remove(selected)
            routes.insert(0, selected)
        visible_routes = routes[:_MAX_VISIBLE_ROUTES]
        for route in visible_routes:
            if route == selected:
                lines.append(f"[green]●[/green] {escape(route)} [dim]active[/dim]")
            else:
                lines.append(f"[dim]•[/dim] {escape(route)}")
        hidden_count = len(routes) - len(visible_routes)
        if hidden_count:
            lines.append(f"[dim]… {hidden_count} more[/dim]")
        if selected is not None and selected not in metadata.routes:
            lines.append(f"[yellow]●[/yellow] {escape(selected)} [dim]not advertised[/dim]")
        return lines

    def _remove_sidebar(self, context: ExtensionContext | None = None) -> None:
        if self._sidebar_render is None:
            return
        self._sidebar_render = None
        sidebar = self._sidebar(context)
        if sidebar is None:
            return
        remove = getattr(sidebar, "remove_section", None)
        if remove is not None:
            remove(_SIDEBAR_KEY)

    def _show_sidebar(self, metadata: _ProviderMetadata | None) -> bool:
        sidebar = self._sidebar()
        if sidebar is None or not bool(getattr(sidebar, "supported", False)):
            return False
        set_section = getattr(sidebar, "set_section", None)
        if set_section is None:
            return False
        title = "hugging face"
        content = tuple(self._content(metadata))
        render = (title, content)
        if render == self._sidebar_render:
            return True
        set_section(_SIDEBAR_KEY, title=title, content=content)
        self._sidebar_render = render
        return True

    def _sync_sidebar(self) -> None:
        context = self.api.context
        if context.provider_name != "huggingface":
            self._remove_sidebar()
            return
        metadata = self._fresh_metadata(context.model)
        if not self._show_sidebar(metadata):
            return
        if metadata is None:
            self._start_sidebar_load(context.model)

    def _fresh_metadata(self, model: str) -> _ProviderMetadata | None:
        metadata = self._metadata
        if metadata is None or metadata.model != model:
            return None
        if time.monotonic() - metadata.fetched_at >= _METADATA_TTL_SECONDS:
            return None
        return metadata

    def _start_sidebar_load(self, model: str) -> None:
        if self._sidebar_task is not None and not self._sidebar_task.done():
            return

        async def load() -> None:
            metadata = await self._get_metadata(model)
            context = self.api.context
            if context.provider_name == "huggingface" and context.model == model:
                self._show_sidebar(metadata)

        self._sidebar_task = asyncio.create_task(load())

    async def _get_metadata(self, model: str) -> _ProviderMetadata:
        cached = self._fresh_metadata(model)
        if cached is not None:
            return cached

        if self._metadata_task is not None and not self._metadata_task.done():
            if self._metadata_model == model:
                return await asyncio.shield(self._metadata_task)
            self._metadata_task.cancel()

        async def fetch() -> _ProviderMetadata:
            try:
                routes = await asyncio.to_thread(_fetch_available_routes, model)
                error = None
            except Exception as exc:  # noqa: BLE001 - metadata is optional
                routes = ()
                error = str(exc)
            return _ProviderMetadata(model, routes, error, time.monotonic())

        task = asyncio.create_task(fetch())
        self._metadata_task = task
        self._metadata_model = model
        metadata = await asyncio.shield(task)
        if self._metadata_task is task:
            self._metadata = metadata
        return metadata

    def _invalidate_for_model_change(self, model: str) -> None:
        if self._metadata is not None and self._metadata.model != model:
            self._metadata = None
        if (
            self._metadata_task is not None
            and not self._metadata_task.done()
            and self._metadata_model != model
        ):
            self._metadata_task.cancel()
        if self._sidebar_task is not None and not self._sidebar_task.done():
            self._sidebar_task.cancel()
        self._sidebar_task = None

    def _clear_response_route_for_state_change(
        self,
        state: tuple[str, str, str, str | None],
    ) -> None:
        previous = self._observed_state
        if previous is None:
            return
        provider_or_model_changed = state[:2] != previous[:2]
        mode_changed = state[2] != previous[2]
        automatic_route_cleared = (
            state[2] == "automatic" and previous[3] is not None and state[3] is None
        )
        if provider_or_model_changed or mode_changed or automatic_route_cleared:
            self._automatic_response_route = None

    def _observe_response_route(self, event: object) -> bool:
        context = self.api.context
        if context.provider_name != "huggingface" or context.inference_provider_mode != "automatic":
            return False
        message = getattr(event, "message", None)
        if message is None or getattr(message, "stop_reason", None) == "error":
            return False
        route = getattr(message, "response_provider", None)
        if not isinstance(route, str) or not route.strip():
            return False
        route = route.strip()
        if route == self._automatic_response_route:
            return False
        self._automatic_response_route = route
        return True

    async def _observe_state(self) -> None:
        while True:
            await asyncio.sleep(_STATE_POLL_SECONDS)
            state = self._state()
            if state == self._observed_state:
                if state[0] == "huggingface" and self._fresh_metadata(state[1]) is None:
                    self._sync_sidebar()
                continue
            self._clear_response_route_for_state_change(state)
            if self._observed_state is None or state[1] != self._observed_state[1]:
                self._invalidate_for_model_change(state[1])
            self._observed_state = state
            self._sync_sidebar()

    def _on_session_start(self, event: object, context: ExtensionContext) -> None:
        del event, context
        self._cancel_tasks()
        self._automatic_response_route = None
        self._sidebar_render = None
        self._observed_state = self._state()
        self._sync_sidebar()
        sidebar = self._sidebar()
        if sidebar is not None and bool(getattr(sidebar, "supported", False)):
            self._observer_task = asyncio.create_task(self._observe_state())

    async def _on_session_shutdown(self, event: object, context: ExtensionContext) -> None:
        del event
        self._remove_sidebar(context)
        tasks = self._cancel_tasks()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _on_session_state_event(self, event: object, context: ExtensionContext) -> None:
        del context
        state = self._state()
        self._clear_response_route_for_state_change(state)
        response_route_changed = self._observe_response_route(event)
        if state == self._observed_state:
            if response_route_changed or (
                state[0] == "huggingface" and self._fresh_metadata(state[1]) is None
            ):
                self._sync_sidebar()
            return
        if self._observed_state is None or state[1] != self._observed_state[1]:
            self._invalidate_for_model_change(state[1])
        self._observed_state = state
        self._sync_sidebar()

    def _cancel_tasks(self) -> tuple[asyncio.Task[object], ...]:
        tasks = tuple(
            task
            for task in (self._observer_task, self._sidebar_task, self._metadata_task)
            if task is not None and not task.done()
        )
        for task in tasks:
            task.cancel()
        self._observer_task = None
        self._sidebar_task = None
        self._metadata_task = None
        self._metadata_model = None
        return tasks

    async def _choose_route(self) -> None:
        """Open the route picker and apply the selected provider."""
        try:
            metadata = await self._get_metadata(self.api.context.model)
            if metadata.error is not None:
                self.api.notify(
                    f"Could not load available Hugging Face providers: {metadata.error}",
                    "warning",
                )

            selected = await self.api.context.ui.select(
                "Hugging Face inference provider",
                ("automatic", *metadata.routes),
            )
            if selected is None:
                return

            selected_route = None if selected.casefold() in _ROUTE_RESET_ALIASES else selected
            self._automatic_response_route = None
            message = self.api.set_inference_provider(selected_route)
            self._observed_state = self._state()
            self._sync_sidebar()
            self.api.notify(f"Hugging Face route: {message}")
        except Exception as exc:  # noqa: BLE001 - background command tasks must not leak errors
            try:
                self.api.notify(f"Could not change Hugging Face route: {exc}", "error")
            except Exception:  # noqa: BLE001 - stale APIs cannot be notified
                return

    def _route(self, args: str, context: ExtensionCommandContext) -> str | None:
        api = context.api
        if api.context.provider_name != "huggingface":
            return "/hf route requires the huggingface provider"

        value = args.strip()
        if not value:
            if not api.context.has_ui:
                return _current_route(api)
            try:
                asyncio.get_running_loop().create_task(self._choose_route())
            except RuntimeError:
                return _current_route(api)
            return None

        selected_route = None if value.casefold() in _ROUTE_RESET_ALIASES else value
        self._automatic_response_route = None
        selected = api.set_inference_provider(selected_route)
        self._observed_state = self._state()
        self._sync_sidebar()
        return f"Hugging Face route: {selected}"

    def _hf(self, args: str, context: ExtensionCommandContext) -> str | None:
        value = args.strip()
        if not value:
            return "Usage: /hf route [automatic|<inference-provider>]"

        command, _, command_args = value.partition(" ")
        if command.casefold() != "route":
            return f"Unknown /hf command: {command}. Available commands: route"
        return self._route(command_args, context)


def setup(tau: ExtensionAPI) -> None:
    """Register Hugging Face-specific commands and sidebar status."""
    _HuggingFaceExtension(tau).register()
