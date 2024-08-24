# LLMLib

LLMLib is a simple Python library for making requests to Language Learning Models (LLMs) such as OpenAI's GPT and Anthropic's Claude.

## Installation

You can install LLMLib using pip:

```
pip install llmlib-amq
```

## Usage

Here's a quick example of how to use LLMLib:

```python
from llmlib import LLMClient, LLMRequest, Provider, OpenAIModel, Message

client = LLMClient(
    openai_api_key="your-openai-api-key",
    anthropic_api_key="your-anthropic-api-key"
)

request = LLMRequest(
    provider=Provider.OPENAI,
    model=OpenAIModel.GPT_3_5_TURBO,
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me a joke.")
    ],
    stream=True
)

for chunk in client.generate(request):
    print(chunk, end="", flush=True)
```


## License

This project is licensed under the MIT License.