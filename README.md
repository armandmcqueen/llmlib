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
from PIL import Image
from pathlib import Path
from llmlib import LLMClient, Provider, AnthropicModel, OpenAIModel, Role, \
    LLMResponse, TextMessage, ImageMessage, print_stream, encode_image_webp
import os


client = LLMClient(
    provider=Provider.ANTHROPIC,
    model=AnthropicModel.CLAUDE_3_5_SONNET,
    # provider=Provider.OPENAI,
    # model=OpenAIModel.GPT_4O,
    openai_key=os.environ["OPENAI_API_KEY"],
    anthropic_key=os.environ["ANTHROPIC_API_KEY"],
)

#############################################
### Non-streaming example                 ###
#############################################
output: LLMResponse = client.chat(
    messages=[
        TextMessage(content="You are very concise, answering a question in one sentence.", role=Role.SYSTEM),
        TextMessage(content="What is the capital of South Africa?", role=Role.USER)
    ]
)
print(f"[{client.model_id}]")
print(output.content)

#############################################
### Non-streaming image example           ###
#############################################
example_image = Path(__file__).parent.parent / "tests/bird.webp"
img_base64 = encode_image_webp(Image.open(example_image))
output: LLMResponse = client.chat(
    messages=[
        TextMessage(content="You are very concise image analyst", role=Role.SYSTEM),
        TextMessage(content="Please describe this image", role=Role.USER),
        ImageMessage(content=img_base64, role=Role.USER)
    ]
)
print(f"[{client.model_id}]")
print(output.content)
print(output.usage)

#############################################
### Streaming example                     ###
#############################################
for chunk in client.chat_stream(
        messages=[TextMessage(content="How can I solve tic-tac-toe in Python?", role=Role.USER)]
):
    print(chunk, end="", flush=True)

#############################################
### Streaming and collecting output       ###
#############################################
stream = client.chat_stream(
    messages=[
        TextMessage(
            content="What is the bash command to rollback a git commit? "
            "Please be extremely succinct, output only the bash "
            "command. Do not include any commentary. The output "
            "will be executed directly",
            role=Role.USER,
        )
    ]
)
result = print_stream(stream)
print()
print(result)

```


## License

This project is licensed under the MIT License.