import pytest
import os
from llmlib import (
    LLMClient,
    Provider,
    OpenAIModel,
    AnthropicModel,
    TextMessage,
    ImageMessage,
    Role,
    LLMResponse,
    process_and_collect_stream,
    print_stream,
    encode_image_webp,
)
from PIL import Image


@pytest.fixture
def openai_client():
    return LLMClient(
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT_4O,
        openai_key=os.environ.get("OPENAI_API_KEY"),
    )


@pytest.fixture
def anthropic_client():
    return LLMClient(
        provider=Provider.ANTHROPIC,
        model=AnthropicModel.CLAUDE_3_5_SONNET,
        anthropic_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def test_chat_openai(openai_client):
    messages = [TextMessage(content="What is the capital of France?", role=Role.USER)]
    response = openai_client.chat(messages)

    assert isinstance(response, LLMResponse)
    assert "Paris" in response.content
    assert response.model.startswith("gpt-4o")
    assert "total_tokens" in response.usage


def test_chat_anthropic(anthropic_client):
    messages = [TextMessage(content="What is the capital of Japan?", role=Role.USER)]
    response = anthropic_client.chat(messages)

    assert isinstance(response, LLMResponse)
    assert "Tokyo" in response.content
    assert response.model.startswith("claude-3-5-sonnet")
    assert "prompt_tokens" in response.usage
    assert "completion_tokens" in response.usage


def test_chat_stream_openai(openai_client):
    messages = [TextMessage(content="Count from 1 to 5.", role=Role.USER)]
    stream = openai_client.chat_stream(messages)
    result = process_and_collect_stream(stream)

    assert all(str(i) in result for i in range(1, 6))


def test_chat_stream_anthropic(anthropic_client):
    messages = [TextMessage(content="List the days of the week.", role=Role.USER)]
    stream = anthropic_client.chat_stream(messages)
    result = process_and_collect_stream(stream)

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    assert all(day in result for day in days)


def test_print_stream(anthropic_client, capsys):
    messages = [TextMessage(content="Say 'Hello, World!'", role=Role.USER)]
    stream = anthropic_client.chat_stream(messages)
    result = print_stream(stream)
    captured = capsys.readouterr()

    assert "Hello, World!" in result
    assert "Hello, World!" in captured.out

def test_object_classification(anthropic_client):
    img_dog = encode_image_webp(Image.open("tests/dog.webp"))
    img_bird = encode_image_webp(Image.open("tests/bird.webp"))
    base_messages = [
        TextMessage(content="You are an image analyst", role=Role.SYSTEM),
        TextMessage(
            content="Is this a dog, cat, or bird? Give a one word response. Do not include punctuation.",
            role=Role.USER
        ),
    ]

    response = anthropic_client.chat(base_messages + [ImageMessage(content=img_dog, role=Role.USER)])
    choice = response.content.lower().strip()
    assert choice == "dog", f"Expected 'dog', got '{choice}'"

    response = anthropic_client.chat(base_messages + [ImageMessage(content=img_bird, role=Role.USER)])
    choice = response.content.lower().strip()
    assert choice == "bird", f"Expected 'bird', got '{choice}'"

