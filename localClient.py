from base import BaseLLMClient
from datetime import datetime, timedelta

MAX_RETRY = 2
MODEL = "qwen3:8b"


class LocalClient(BaseLLMClient):
    def __init__(self):
        try:
            from ollama import Client

            self.client = Client(host="http://127.0.0.1:11434")
            self.rateLimited = False
        except Exception as e:
            print(f"Local client failed to initialize: {e}")
            raise

    def generate(self, messages, **kwargs):
        for attempt in range(MAX_RETRY):
            try:
                options = kwargs.pop("options", {"temperature": 0.1})
                response = self.client.chat(
                    model=MODEL,
                    messages=messages,
                    stream=False,
                    options=options,
                    **kwargs,
                )
                msg = response.message
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
                print(f"Ollama failed to generate: {e}")
                break
        self.rateLimited = True
        return False

    def close(self):
        self.client.close()
