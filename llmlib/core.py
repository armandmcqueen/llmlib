from enum import Enum
from typing import Generator, List, Optional, Union

from pydantic import BaseModel, Field
import openai
from anthropic import Anthropic


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class OpenAIModel(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"


class AnthropicModel(Enum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"


Model = Union[OpenAIModel, AnthropicModel]


class Message(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    provider: Provider
    model: Model
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = False


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict


class LLMClient:
    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

    def generate(self, request: LLMRequest) -> Union[LLMResponse, Generator[str, None, None]]:
        if request.provider == Provider.OPENAI:
            return self._generate_openai(request)
        elif request.provider == Provider.ANTHROPIC:
            return self._generate_anthropic(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

    def _generate_openai(self, request: LLMRequest) -> Union[LLMResponse, Generator[str, None, None]]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            def stream_generator() -> Generator[str, None, None]:
                for chunk in self.openai_client.chat.completions.create(
                        model=request.model.value,
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True
                ):
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return stream_generator()
        else:
            response = self.openai_client.chat.completions.create(
                model=request.model.value,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump()
            )

    def _generate_anthropic(self, request: LLMRequest) -> Union[LLMResponse, Generator[str, None, None]]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            def stream_generator() -> Generator[str, None, None]:
                with self.anthropic_client.messages.stream(
                        model=request.model.value,
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                ) as stream:
                    for text in stream.text_iterator():
                        yield text

            return stream_generator()
        else:
            response = self.anthropic_client.messages.create(
                model=request.model.value,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )


def main():
    client = LLMClient(
        openai_api_key="your-openai-api-key",
        anthropic_api_key="your-anthropic-api-key"
    )

    # Example usage with OpenAI
    openai_request = LLMRequest(
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT_3_5_TURBO,
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Tell me a joke.")
        ],
        stream=True
    )

    # Example usage with Anthropic
    anthropic_request = LLMRequest(
        provider=Provider.ANTHROPIC,
        model=AnthropicModel.CLAUDE_3_SONNET,
        messages=[
            Message(role="human", content="Tell me a joke.")
        ],
        stream=False
    )

    # Streaming example
    for chunk in client.generate(openai_request):
        print(chunk, end="", flush=True)
    print("\n")

    # Non-streaming example
    response = client.generate(anthropic_request)
    print(response.content)


if __name__ == "__main__":
    main()