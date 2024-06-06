from src.utils import Config
from .preprocessor import Preprocessor

def get_preprocessor(config: Config):
    preprocessors = {
        'base': Preprocessor
    }

    return preprocessors[config.preprocessor.name](config.preprocessor)