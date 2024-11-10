from enum import Enum
from typing import Optional, Generator, Union, Iterable, Callable, cast, TypedDict, Literal
from abc import ABC

from anthropic.types import TextBlockParam, ImageBlockParam
from pydantic import BaseModel
from typing_extensions import TypeAlias
import openai
from anthropic import Anthropic
import anthropic


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class OpenAIModel(StrEnum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"


class AnthropicModel(StrEnum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"


Model = Union[OpenAIModel, AnthropicModel]


class ChatCompletionSystemMessageParam(TypedDict):
    role: str
    content: str


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel, ABC):
    content: str
    role: Role


class TextMessage(Message):
    content: str
    role: Role


class ImageMessage(Message):
    content: str  # base64 encoded webp
    role: Role


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict  # TODO: Type this properly


_OpenAiChatCompletionMessageParam: TypeAlias = Union[
    openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam,
    openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam,
    openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam,
    openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam,
    openai.types.chat.chat_completion_function_message_param.ChatCompletionFunctionMessageParam,
]


def _convert_messages_to_openai_types(
        messages: list[Message],
) -> list[_OpenAiChatCompletionMessageParam]:
    o: list[_OpenAiChatCompletionMessageParam] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            o.append(
                openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam(
                    role="system", content=m.content
                )
            )
        elif m.role == Role.USER:
            o.append(
                openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam(
                    role="user", content=m.content
                )
            )
        elif m.role == Role.ASSISTANT:
            o.append(
                openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam(
                    role="assistant", content=m.content
                )
            )
        else:
            raise ValueError(f"Unsupported role: {m.role}")
    return o


def _anthropic_extract_system_prompt(
        messages: list[Message],
) -> str | anthropic.NotGiven:
    # Confirm there is one or zero system messages
    system_messages = [m for m in messages if m.role == Role.SYSTEM]
    if len(system_messages) > 1:
        raise ValueError("Only one system message is allowed")

    if len(system_messages) == 0:
        return anthropic.NotGiven()
    return system_messages[0].content


def _anthropic_filter_system_messages(messages: list[Message]) -> list[Message]:
    filtered_messages = [m for m in messages if m.role != Role.SYSTEM]
    if len(filtered_messages) == 0:
        raise ValueError("At least one non-system message is required for Anthropic")
    return filtered_messages


def _anthropic_convert_messages_to_typed_dicts(
        messages: list[Message],
) -> list[anthropic.types.message_param.MessageParam]:
    o: list[anthropic.types.message_param.MessageParam] = []
    # First, we need to iterate over the messages and collect contiguous assistant and user message
    # so we can combine them. We must also validate that there are no system messages and that
    # we always alternate between assistant and user messages.
    # [msg1, msg2, msg3, msg4, msg5] -> [[msg1, msg2], [msg3], [msg4, msg5]]
    contiguous_messages: list[list[Message]] = []
    current_messages: list[Message] = []
    current_role = messages[0].role.USER

    for m in messages:
        if m.role == Role.SYSTEM:
            raise ValueError("System messages are not supported in Anthropic")

        # Check if we are switching roles
        if m.role != current_role:
            contiguous_messages.append(current_messages)
            current_messages = []
            current_role = m.role

        current_messages.append(m)

    contiguous_messages.append(current_messages)

    for cm in contiguous_messages:
        role: Literal["user", "assistant"] = "user" if cm[0].role == Role.USER else "assistant"
        content: list[TextBlockParam | ImageBlockParam] = []
        for m in cm:
            if isinstance(m, TextMessage):
                content.append({"type": "text", "text": m.content})
            elif isinstance(m, ImageMessage):
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/webp",
                            "data": m.content
                        }
                    }
                )
            else:
                raise ValueError(f"Unsupported message type: {m}")
        o.append(anthropic.types.message_param.MessageParam(role=role, content=content))

    return o


def _anthropic_filter_and_convert_messages_to_typed_dicts(
        messages: list[Message],
) -> list[anthropic.types.message_param.MessageParam]:
    return _anthropic_convert_messages_to_typed_dicts(
        _anthropic_filter_system_messages(messages)
    )


class LLMClient:
    def __init__(
            self,
            provider: Provider,
            model: Model,
            openai_key: Optional[str] = None,
            anthropic_key: Optional[str] = None,
            anthropic_max_tokens: int = 8192,
    ):
        self.provider = provider
        self.model = model
        self.anthropic_max_tokens = anthropic_max_tokens
        self.openai_client = None
        self.anthropic_client = None

        if provider == Provider.OPENAI:
            if not openai_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            self.openai_client = openai.OpenAI(api_key=openai_key)
        elif provider == Provider.ANTHROPIC:
            if not anthropic_key:
                raise ValueError("Anthropic API key is required for Anthropic provider")
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def model_id(self) -> str:
        return f"{self.provider.value}/{self.model.value}"

    def chat(self, messages: list[Message]) -> LLMResponse:

        if self.provider == Provider.OPENAI:
            if self.openai_client is None:
                raise ValueError("OpenAI client is not initialized")

            for m in messages:
                if isinstance(m, ImageMessage):
                    raise ValueError("Image messages are not supported in OpenAI")

            openai_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=_convert_messages_to_openai_types(messages),
            )
            content_response = openai_response.choices[0].message.content
            if (
                    content_response is None
                    or openai_response.model is None
                    or openai_response.usage is None
            ):
                raise ValueError("Invalid response from OpenAI")
            return LLMResponse(
                content=content_response,
                model=openai_response.model,
                usage=openai_response.usage.model_dump(),
            )
        elif self.provider == Provider.ANTHROPIC:
            if self.anthropic_client is None:
                raise ValueError("Anthropic client is not initialized")
            # TODO: I have no idea how to make this type crap work. Anthropic is a mess.

            anthropic_response = self.anthropic_client.messages.create(
                model=self.model,
                messages=_anthropic_filter_and_convert_messages_to_typed_dicts(
                    messages
                ),
                max_tokens=self.anthropic_max_tokens,
                system=_anthropic_extract_system_prompt(messages),
            )
            content_response = anthropic_response.content[0].text  # type: ignore
            if (
                    content_response is None
                    or anthropic_response.model is None
                    or anthropic_response.usage is None
            ):
                raise ValueError("Invalid response from Anthropic")
            if anthropic_response.content[0].type == "tool_result":
                raise NotImplementedError(
                    "Anthropic ToolResultBlockParam (tool calling) is not supported"
                )
            content = anthropic_response.content[0]
            content = cast(anthropic.types.TextBlock, content)
            return LLMResponse(
                content=content.text,
                model=anthropic_response.model,
                usage={
                    "prompt_tokens": anthropic_response.usage.input_tokens,
                    "completion_tokens": anthropic_response.usage.output_tokens,
                    "total_tokens": anthropic_response.usage.input_tokens
                                    + anthropic_response.usage.output_tokens,
                },
            )

    def chat_stream(self, messages: list[Message]) -> Generator[str, None, None]:

        if self.provider == Provider.OPENAI:
            if self.openai_client is None:
                raise ValueError("OpenAI client is not initialized")
            openai_stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=_convert_messages_to_openai_types(messages),
                stream=True,
            )
            openai_stream = cast(openai.Stream, openai_stream)
            for chunk in openai_stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        elif self.provider == Provider.ANTHROPIC:
            if self.anthropic_client is None:
                raise ValueError("Anthropic client is not initialized")
            with self.anthropic_client.messages.stream(  # type: ignore
                    model=self.model,
                    messages=_anthropic_filter_and_convert_messages_to_typed_dicts(
                        messages
                    ),
                    max_tokens=self.anthropic_max_tokens,
                    system=_anthropic_extract_system_prompt(messages),
            ) as anthropic_stream:
                for text in anthropic_stream.text_stream:
                    yield text


def process_and_collect_stream(
        stream: Iterable[str],
        chunk_fn: Callable[[str], None] = lambda x: print(x, end="", flush=True),
) -> str:
    """
    Process a stream of text chunks, applying an optional function to each chunk,
    and return the complete output.

    Args:
        stream (Iterable[str]): An iterable of text chunks.
        chunk_fn (Optional[Callable[[str], None]]): A function to apply to each chunk.

    Returns:
        str: The complete output as a single string.
    """
    full_output = ""
    for chunk in stream:
        full_output += chunk
        chunk_fn(chunk)

    return full_output


def print_stream(stream: Iterable[str]) -> str:
    """
    Print a stream of text chunks.

    Args:
        stream (Iterable[str]): An iterable of text chunks.

    Returns:
        str: The complete output as a single string.
    """
    return process_and_collect_stream(stream)

