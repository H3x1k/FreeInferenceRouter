import os
from dotenv import load_dotenv

from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

from localClient import LocalClient
from geminiClient import GeminiClient
from groqClient import GroqClient
from mistralClient import MistralClient


class Router:
    def __init__(self):
        self.clients = [
            GroqClient(GROQ_API_KEY),
            MistralClient(MISTRAL_API_KEY),
            GeminiClient(GEMINI_API_KEY),
        ]
        try:
            localClient = LocalClient()
            self.clients.append(localClient)
        except:
            pass

    def generate(self, messages, **kwargs):
        for client in self.clients:
            if client.retryAt and datetime.now() > client.retryAt:
                client.rateLimited = False
                client.retryAt = None
            if not client.rateLimited:
                response = client.generate(messages=messages, **kwargs)
                if response:
                    return response
        print("NO AVAILABLE PROVIDER")
        for client in self.clients:
            print(client.rateLimited, client.retryAt)
