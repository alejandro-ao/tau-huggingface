"""Hugging Face-specific commands for Tau."""

from tau_coding.extensions import ExtensionAPI, ExtensionCommandContext


def _route(args: str, context: ExtensionCommandContext) -> str:
    api = context.api
    if api.context.provider_name != "huggingface":
        return "/route requires the huggingface provider"

    value = args.strip()
    if not value:
        route = api.context.inference_provider or "automatic"
        return f"Hugging Face route: {route}"

    selected_route = None if value.casefold() in {"automatic", "auto", "reset"} else value
    selected = api.set_inference_provider(selected_route)
    return f"Hugging Face route: {selected}"


def setup(tau: ExtensionAPI) -> None:
    """Register Hugging Face-specific commands."""
    tau.register_command(
        "route",
        _route,
        description="Show or change Hugging Face session routing.",
        usage="/route [automatic|<inference-provider>]",
    )
