from src.utils import Config
from .chatgpt_reasoner import ChatGPTReasoner

def get_reasoner(config: Config):
    reasoners = {
        'chatgpt': ChatGPTReasoner
    }

    return reasoners[config.reasoner.name](config.reasoner)