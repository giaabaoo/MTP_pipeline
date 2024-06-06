from src.utils import Config
from .whisperx_segmentor import WhisperXSegmentor

def get_segmentor(config: Config):
    segmentors = {
        'whisperx': WhisperXSegmentor
    }

    return segmentors[config.segmentor.name](config.segmentor)