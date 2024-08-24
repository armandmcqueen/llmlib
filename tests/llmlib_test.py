# tests/test_llmlib.py
import unittest
from unittest.mock import patch, MagicMock
from llmlib import LLMClient, LLMRequest, Provider, OpenAIModel, Message

class TestLLMLib(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(openai_api_key="fake-openai-key", anthropic_api_key="fake-anthropic-key")

    @patch('openai.OpenAI')
    def test_openai_generate(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))],
            model="gpt-3.5-turbo",
            usage=MagicMock(model_dump=lambda: {"total_tokens": 10})
        )

        request = LLMRequest(
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT_3_5_TURBO,
            messages=[Message(role="user", content="Test message")]
        )

        response = self.client.generate(request)

        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "gpt-3.5-turbo")
        self.assertEqual(response.usage, {"total_tokens": 10})

if __name__ == '__main__':
    unittest.main()