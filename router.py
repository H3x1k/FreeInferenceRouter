import os
from dotenv import load_dotenv

from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


from localClient import LocalClient
from gemini import GeminiClient
from groqClient import GroqClient


class Router:
    def __init__(self):
        self.clients = [
            GroqClient(GROQ_API_KEY),
            GeminiClient(GEMINI_API_KEY)
        ]
        try:
            localClient = LocalClient()
            self.clients.extend(localClient)
        except:
            pass

    def generate(self, messages, **kwargs):
        for client in self.clients:
            if client.retryAt and datetime.now() > client.retryAt:
                client.rateLimited = False
                client.retryAt = None
            if not client.rateLimited:
                response = client.generate(messages=messages, kwargs=kwargs)
                print(response)
                if response:
                    return response
        print("NO AVAILABLE PROVIDER")
        for client in self.clients:
            print(client.rateLimited, client.retryAt)