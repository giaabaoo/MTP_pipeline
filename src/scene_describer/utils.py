from src.utils import Config
from .llava_scene_describer import LLAVASceneDescriber
from .gpt4v_scene_describer import GPT4VSceneDescriber
from src.scene_describer.frames_sampler.base_frames_sampler import BaseFramesSampler
from src.scene_describer.frames_sampler.sparse_frames_sampler import SparseFramesSampler


def get_scene_describer(config: Config):
    scene_describers = {
        'LLAVA': LLAVASceneDescriber,
        'GPT4V': GPT4VSceneDescriber
    }

    return scene_describers[config.scene_describer.name](config.scene_describer)

def get_frames_sampler(config: Config):
    frames_samplers = {
        'base': BaseFramesSampler,
        'sparse': SparseFramesSampler
    }
    
    return frames_samplers[config.frames_sampler.name](config.frames_sampler)