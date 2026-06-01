from base import BaseLLMClient
from ollama import Client

from datetime import datetime, timedelta

MAX_RETRY = 2
MODEL = "qwen3:8b"

class LocalClient(BaseLLMClient):
    def __init__(self):
        try:
            self.client = Client(host="http://127.0.0.1:11434")
            self.rateLimited = False
        except Exception as e:
            print(f"Local client failed to initialize: {e}")
    
    def generate(self, messages, **kwargs):
        for attempt in range(MAX_RETRY):
            try:
                response = self.client.chat(
                    model=MODEL,
                    messages=messages,
                    stream=False,
                    options={"temperature": 0.1}
                )
                return response.message.content
            except Exception as e:
                print(f"Ollama failed to generate: {e}")
                break
        self.rateLimited = True
        return False
        
    def close(self):
        self.client.close()