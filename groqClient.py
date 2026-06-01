from base import BaseLLMClient
from groq import Groq, RateLimitError, APIError

from datetime import datetime, timedelta

MAX_RETRY = 2

class GroqClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.rateLimited = False
        self.retryAt = None
    
    def generate(self, messages, **kwargs):
        for attempt in range(MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                print("Groq rate limited")
                self.rateLimited = True
                self.retryAt = datetime.now() + timedelta(seconds=60)
                return False
            except APIError as e:
                print(f"Groq API Error {e.status_code}: {e.message}")
                return False 
        return False
        
    def close(self):
        self.client.close()