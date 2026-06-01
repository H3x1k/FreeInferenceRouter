from abc import ABC, abstractmethod

class BaseLLMClient(ABC):

    @classmethod
    @abstractmethod
    def generate(self, messages: list, **kwargs):
        """Non-stream inference"""
        pass

#   def stream_generate(self, messages: list, **kwargs)
#       """Streaming inference"""
#       pass

    @classmethod
    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass