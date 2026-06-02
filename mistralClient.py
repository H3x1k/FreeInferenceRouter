from mistralai.client import Mistral
from base import BaseLLMClient
from datetime import datetime, timedelta

MAX_RETRY = 2


class MistralClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        self.rateLimited = False
        self.retryAt = None

    def generate(self, messages, **kwargs):
        for attempt in range(MAX_RETRY):
            try:
                response = self.client.chat.complete(
                    model="mistral-large-latest", messages=messages, **kwargs
                )
                msg = response.choices[0].message
                tool_calls = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                return {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }
            except Exception as e:
                print("Mistral error")
                self.rateLimited = True
                self.retryAt = datetime.now() + timedelta(seconds=60)
                break
        return False

    def close(self):
        self.client.close()
