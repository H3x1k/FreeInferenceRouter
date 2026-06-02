from base import BaseLLMClient
from openai import OpenAI, RateLimitError, APIError

from datetime import datetime, timedelta

MAX_RETRY = 2


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.rateLimited = False
        self.retryAt = None

    def generate(self, messages, **kwargs):
        for attempt in range(MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model="gemini-2.5-flash", messages=messages, **kwargs
                )
                msg = response.choices[0].message
                return {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            except RateLimitError as e:
                print("Gemini rate limited")
                self.rateLimited = True
                self.retryAt = datetime.now() + timedelta(seconds=60)
                return False
            except APIError as e:
                print(f"Gemini API Error {e.status_code}: {e.message}")
                return False
        return False

    def close(self):
        self.client.close()
