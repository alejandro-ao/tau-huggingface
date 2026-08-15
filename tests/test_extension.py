from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tau_agent.messages import AgentMessage
from tau_coding.commands import CommandRegistry, CommandResult, CommandSession
from tau_coding.extensions import ExtensionRuntime
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


def _registry(tmp_path: Path) -> tuple[FakeSession, CommandRegistry]:
    runtime = ExtensionRuntime()
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

    assert _execute(registry, session, "/route").message == "Hugging Face route: automatic"
    assert (
        _execute(registry, session, "/route deepinfra").message == "Hugging Face route: deepinfra"
    )
    assert session.inference_provider == "deepinfra"
    assert _execute(registry, session, "/route reset").message == (
        "Hugging Face route: automatic (will pin after the next successful response)"
    )
    assert session.inference_provider is None


def test_route_rejects_other_tau_providers(tmp_path: Path) -> None:
    session, registry = _registry(tmp_path)
    session.provider_name = "openai"

    assert _execute(registry, session, "/route").message == (
        "/route requires the huggingface provider"
    )
