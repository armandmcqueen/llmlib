from enum import Enum
from typing import Optional, Generator, Union, Iterable, Callable, cast, TypedDict

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
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"


Model = Union[OpenAIModel, AnthropicModel]


class ChatCompletionSystemMessageParam(TypedDict):
    role: str
    content: str


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    content: str
    role: Role


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict


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
    for m in messages:
        if m.role == Role.SYSTEM:
            raise ValueError("System messages are not supported in Anthropic")
        elif m.role == Role.USER:
            o.append(
                anthropic.types.message_param.MessageParam(
                    role="user", content=m.content
                )
            )
        elif m.role == Role.ASSISTANT:
            o.append(
                anthropic.types.message_param.MessageParam(
                    role="assistant", content=m.content
                )
            )
        else:
            raise ValueError(f"Unsupported role: {m.role}")
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


if __name__ == "__main__":

    # Example usage
    import os

    client = LLMClient(
        provider=Provider.ANTHROPIC,
        model=AnthropicModel.CLAUDE_3_5_SONNET,
        # provider=Provider.OPENAI,
        # model=OpenAIModel.GPT_4O,
        openai_key=os.environ["OPENAI_API_KEY"],
        anthropic_key=os.environ["ANTHROPIC_API_KEY"],
    )

    # Non-streaming example
    output: LLMResponse = client.chat(
        messages=[
            Message(content="You are very concise, answering a question in one sentence.", role=Role.SYSTEM),
            Message(content="What is the capital of South Africa?", role=Role.USER)
        ]
    )
    print(f"[{client.model_id}]")
    print(output.content)

    # Streaming example
    # for chunk in client.chat_stream(
    #         messages=[Message(content="How can I solve tic-tac-toe in Python?", role=Role.USER)]
    # ):
    #     print(chunk, end="", flush=True)

    # Streaming and collecting output
    # stream = client.chat_stream(
    #     messages=[
    #         Message(
    #             content="What is the bash command to rollback a git commit? "
    #             "Please be extremely succinct, output only the bash "
    #             "command. Do not include any commentary. The output "
    #             "will be executed directly",
    #             role=Role.USER,
    #         )
    #     ]
    # )
    # result = print_stream(stream)
    # print()
    # print(result)
