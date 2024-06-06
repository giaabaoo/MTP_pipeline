from src.utils.config import Config

from .utterance_level_video import UtteranceLevelVideoLoader
from .utterance_level_transcript import UtteranceLevelTranscriptLoader
from .video import VideoLoader

def get_data_loader(config: Config):
    data_loaders = {
        'utterance_level_video': UtteranceLevelVideoLoader,
        'utterance_level_transcript': UtteranceLevelTranscriptLoader,
        'video': VideoLoader
    }

    return data_loaders[config.data.name](config)
