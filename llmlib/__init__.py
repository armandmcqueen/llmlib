"""A simple library for making requests to LLMs."""

__version__ = "0.0.2"  # Update this manually when you release a new version


# ruff: noqa: F401
from .core import (
    LLMClient,
    Provider,
    OpenAIModel,
    AnthropicModel,
    Model,
    Role,
    Message,
    TextMessage,
    ImageMessage,
    LLMResponse,
    process_and_collect_stream,
    print_stream,
)
from .images import (
    encode_image_webp,
    display_image,
    FluxRatio,
    flux_generate,
    parallel_flux_generate
)