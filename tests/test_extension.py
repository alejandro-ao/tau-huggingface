import asyncio
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib import request

import pytest
from tau_agent.messages import AgentMessage
from tau_coding.commands import CommandRegistry, CommandResult, CommandSession
from tau_coding.extensions import ExtensionRuntime, NullUiBridge
from tau_coding.resources import TauResourcePaths

_EXTENSION_NAME = "tau-huggingface"


@dataclass
class FakeSession:
    cwd: Path
    model: str = "zai-org/GLM-5.2"
    provider_name: str = "huggingface"
    inference_provider: str | None = None
    inference_provider_mode: str = "automatic"
    session_id: str | None = "test-session"
    system_prompt: str = "You are Tau."
    is_running: bool = False
    messages: tuple[AgentMessage, ...] = ()

    def set_inference_provider(self, route: str | None) -> str:
        self.inference_provider = route
        self.inference_provider_mode = "fixed" if route is not None else "automatic"
        return route or "automatic (will pin after the next successful response)"


class RecordingUiBridge(NullUiBridge):
    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.selections: list[tuple[str, tuple[str, ...]]] = []
        self.notifications: list[tuple[str, str]] = []
        self.sidebar_sections: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.sidebar_removals: list[tuple[str, str]] = []

    @property
    def has_ui(self) -> bool:
        return True

    @property
    def supports_sidebar(self) -> bool:
        return True

    def set_sidebar_section(
        self,
        extension_name: str,
        key: str,
        *,
        title: str,
        content: object,
    ) -> None:
        assert isinstance(content, Sequence)
        self.sidebar_sections.append(
            (extension_name, key, title, tuple(cast(Sequence[str], content)))
        )

    def remove_sidebar_section(self, extension_name: str, key: str) -> None:
        self.sidebar_removals.append((extension_name, key))

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        del timeout
        self.selections.append((title, tuple(options)))
        return self.selected

    def notify(self, message: str, level: str = "info") -> None:
        self.notifications.append((message, level))


def _registry(
    tmp_path: Path,
    *,
    ui: NullUiBridge | None = None,
) -> tuple[FakeSession, CommandRegistry, ExtensionRuntime]:
    runtime = ExtensionRuntime(ui=ui)
    runtime.load(
        TauResourcePaths(
            root=tmp_path / "tau",
            cwd=tmp_path,
            agents_root=tmp_path / "agents",
        ),
        extra_paths=(Path(__file__).parents[1],),
        include_resource_dirs=False,
    )
    assert runtime.extension_names == (_EXTENSION_NAME,)
    assert runtime.diagnostics == ()
    session = FakeSession(tmp_path)
    runtime.bind(session)  # type: ignore[arg-type]
    return session, runtime.build_command_registry(), runtime


def _execute(registry: CommandRegistry, session: FakeSession, command: str) -> CommandResult:
    return registry.execute(cast(CommandSession, session), command)


def _latest_sidebar_content(ui: RecordingUiBridge) -> tuple[str, ...]:
    return ui.sidebar_sections[-1][-1]


def test_route_inspects_selects_and_resets(tmp_path: Path) -> None:
    session, registry, _ = _registry(tmp_path)

    assert _execute(registry, session, "/hf route").message == "Hugging Face route: automatic"
    assert (
        _execute(registry, session, "/hf route deepinfra").message
        == "Hugging Face route: deepinfra"
    )
    assert session.inference_provider == "deepinfra"
    assert _execute(registry, session, "/hf route reset").message == (
        "Hugging Face route: automatic (will pin after the next successful response)"
    )
    assert session.inference_provider is None


def test_route_opens_provider_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = RecordingUiBridge("deepinfra")

    response = json.dumps(
        {
            "inferenceProviderMapping": {
                "together": {"status": "live"},
                "offline": {"status": "error"},
                "deepinfra": {"status": "live"},
            }
        }
    ).encode()

    def fake_urlopen(url_request: request.Request, *, timeout: float) -> io.BytesIO:
        assert url_request.full_url.endswith(
            "zai-org/GLM-5.2?expand%5B%5D=inferenceProviderMapping"
        )
        assert timeout == 10
        return io.BytesIO(response)

    async def fake_to_thread(function: object, *args: object) -> object:
        return function(*args)  # type: ignore[operator]

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    session, registry, _ = _registry(tmp_path, ui=ui)

    async def run_command() -> None:
        assert _execute(registry, session, "/hf route").message is None
        for _ in range(4):
            await asyncio.sleep(0)

    asyncio.run(run_command())

    assert ui.selections == [
        (
            "Hugging Face inference provider",
            ("automatic", "deepinfra", "together"),
        )
    ]
    assert session.inference_provider == "deepinfra"
    assert ui.notifications == [("Hugging Face route: deepinfra", "info")]


def test_route_rejects_other_tau_providers(tmp_path: Path) -> None:
    session, registry, _ = _registry(tmp_path)
    session.provider_name = "openai"

    assert _execute(registry, session, "/hf route").message == (
        "/hf route requires the huggingface provider"
    )


def test_sidebar_loads_status_and_updates_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = RecordingUiBridge(None)

    async def fake_to_thread(function: object, *args: object) -> object:
        del function
        assert args == ("zai-org/GLM-5.2",)
        return ("deepinfra", "together")

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    session, registry, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)

        assert ui.sidebar_sections[0] == (
            _EXTENSION_NAME,
            "provider-status",
            "hugging face",
            (
                "[b]zai-org/GLM-5.2[/b]",
                "[dim]○ automatic routing[/dim]",
                "[dim]Loading providers…[/dim]",
            ),
        )
        assert ui.sidebar_sections[-1][-1] == (
            "[b]zai-org/GLM-5.2[/b]",
            "[dim]○ automatic routing[/dim]",
            "[dim]available providers[/dim]",
            "[dim]•[/dim] deepinfra",
            "[dim]•[/dim] together",
        )

        assert _execute(registry, session, "/hf route deepinfra").message == (
            "Hugging Face route: deepinfra"
        )
        updated_content = _latest_sidebar_content(ui)
        assert updated_content[1] == "[green]●[/green] fixed via deepinfra"
        assert updated_content[3] == "[green]●[/green] deepinfra [dim]active[/dim]"
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())
    assert ui.sidebar_removals[-1] == (_EXTENSION_NAME, "provider-status")


def test_sidebar_skips_unchanged_events_and_republishes_after_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = RecordingUiBridge(None)

    async def fake_to_thread(function: object, *args: object) -> object:
        del function, args
        return ("deepinfra",)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    session, _, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)

        publication_count = len(ui.sidebar_sections)
        for event_type in ("agent_start", "message_end", "agent_end"):
            await runtime.emit_event(type("Event", (), {"type": event_type})())
        assert len(ui.sidebar_sections) == publication_count

        session.provider_name = "openai"
        await runtime.emit_event(type("Event", (), {"type": "agent_start"})())
        await runtime.emit_event(type("Event", (), {"type": "message_end"})())
        assert ui.sidebar_removals == [(_EXTENSION_NAME, "provider-status")]

        session.provider_name = "huggingface"
        await runtime.emit_event(type("Event", (), {"type": "agent_start"})())
        assert len(ui.sidebar_sections) == publication_count + 1
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())


def test_sidebar_tracks_successful_automatic_response_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = RecordingUiBridge(None)

    async def fake_to_thread(function: object, *args: object) -> object:
        del function, args
        return ("deepinfra", "together")

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    session, registry, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)

        publication_count = len(ui.sidebar_sections)
        message = type(
            "Message",
            (),
            {"response_provider": "deepinfra", "stop_reason": "stop"},
        )()
        event = type("Event", (), {"type": "message_end", "message": message})()
        await runtime.emit_event(event)

        content = _latest_sidebar_content(ui)
        assert content[1] == "[green]●[/green] automatic via deepinfra"
        assert content[3] == "[green]●[/green] deepinfra [dim]active[/dim]"
        assert len(ui.sidebar_sections) == publication_count + 1

        await runtime.emit_event(event)
        assert len(ui.sidebar_sections) == publication_count + 1

        assert _execute(registry, session, "/hf route together").message == (
            "Hugging Face route: together"
        )
        assert _latest_sidebar_content(ui)[1] == "[green]●[/green] fixed via together"
        assert _execute(registry, session, "/hf route automatic").message == (
            "Hugging Face route: automatic (will pin after the next successful response)"
        )
        assert _latest_sidebar_content(ui)[1] == "[dim]○ automatic routing[/dim]"

        await runtime.emit_event(event)
        session.model = "meta-llama/Llama-4"
        await runtime.emit_event(type("Event", (), {"type": "agent_start"})())
        assert _latest_sidebar_content(ui)[1] == "[dim]○ automatic routing[/dim]"
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())


def test_sidebar_refreshes_for_model_change_and_hides_for_other_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = RecordingUiBridge(None)
    fetched_models: list[str] = []

    async def fake_to_thread(function: object, *args: object) -> object:
        del function
        model = cast(str, args[0])
        fetched_models.append(model)
        return ("provider-for-new-model",)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    session, _, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)

        session.model = "meta-llama/Llama-4"
        await runtime.emit_event(type("Event", (), {"type": "agent_start"})())
        for _ in range(4):
            await asyncio.sleep(0)

        assert fetched_models == ["zai-org/GLM-5.2", "meta-llama/Llama-4"]
        assert ui.sidebar_sections[-1][-1][0] == "[b]meta-llama/Llama-4[/b]"

        session.provider_name = "openai"
        await runtime.emit_event(type("Event", (), {"type": "agent_start"})())
        assert ui.sidebar_removals[-1] == (_EXTENSION_NAME, "provider-status")
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())


def test_sidebar_shows_metadata_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = RecordingUiBridge(None)

    async def failing_to_thread(function: object, *args: object) -> object:
        del function, args
        raise OSError("offline")

    monkeypatch.setattr(asyncio, "to_thread", failing_to_thread)
    _, _, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)
        assert ui.sidebar_sections[-1][-1][-1] == "[yellow]Providers unavailable[/yellow]"
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())


def test_sidebar_shows_when_no_live_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = RecordingUiBridge(None)

    async def fake_to_thread(function: object, *args: object) -> object:
        del function, args
        return ()

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    _, _, runtime = _registry(tmp_path, ui=ui)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        for _ in range(4):
            await asyncio.sleep(0)
        assert ui.sidebar_sections[-1][-1][-1] == "[dim]No live providers[/dim]"
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())


def test_sidebar_is_optional_without_supported_host(tmp_path: Path) -> None:
    session, registry, runtime = _registry(tmp_path)

    async def run() -> None:
        await runtime.emit_session_start("startup")
        assert _execute(registry, session, "/hf route deepinfra").message == (
            "Hugging Face route: deepinfra"
        )
        await runtime.emit_session_shutdown("quit")

    asyncio.run(run())
