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


@dataclass
class FakeSession:
    cwd: Path
    model: str = "zai-org/GLM-5.2"
    provider_name: str = "huggingface"
    inference_provider: str | None = None
    session_id: str | None = "test-session"
    system_prompt: str = "You are Tau."
    is_running: bool = False
    messages: tuple[AgentMessage, ...] = ()

    def set_inference_provider(self, route: str | None) -> str:
        self.inference_provider = route
        return route or "automatic (will pin after the next successful response)"


class RecordingUiBridge(NullUiBridge):
    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.selections: list[tuple[str, tuple[str, ...]]] = []
        self.notifications: list[tuple[str, str]] = []

    @property
    def has_ui(self) -> bool:
        return True

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
) -> tuple[FakeSession, CommandRegistry]:
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
    assert runtime.extension_names == ("huggingface",)
    assert runtime.diagnostics == ()
    session = FakeSession(tmp_path)
    runtime.bind(session)  # type: ignore[arg-type]
    return session, runtime.build_command_registry()


def _execute(registry: CommandRegistry, session: FakeSession, command: str) -> CommandResult:
    return registry.execute(cast(CommandSession, session), command)


def test_route_inspects_selects_and_resets(tmp_path: Path) -> None:
    session, registry = _registry(tmp_path)

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
    session, registry = _registry(tmp_path, ui=ui)

    async def run_command() -> None:
        assert _execute(registry, session, "/hf route").message is None
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
    session, registry = _registry(tmp_path)
    session.provider_name = "openai"

    assert _execute(registry, session, "/hf route").message == (
        "/hf route requires the huggingface provider"
    )
