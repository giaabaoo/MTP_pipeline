from src.utils import Config
from .chatgpt_trainer import ChatGPTTrainer

def get_trainer(config: Config):
    trainers = {
        'gpt-3.5-turbo': ChatGPTTrainer
    }

    return trainers[config.trainer.name](config.trainer)