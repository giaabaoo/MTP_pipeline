from src.utils import Config
from .non_segmented_inference import NonSegmentedInference
from .non_segmented_scene_describer_inference import NonSegmentedSceneDescriberInference
from .non_segmented_summarizer_inference import NonSegmentedSummarizerInference
from .transcript_only_inference import TranscriptOnlyInference
from .inference import Inference
from .train import Train
from .evaluate import Evaluate
from .preprocess import Preprocess
from .extract import Extract

def get_pipeline(config: Config):
    pipelines = {
        'non_segmented_inference': NonSegmentedInference,
        'non_segmented_scene_describer_inference': NonSegmentedSceneDescriberInference,
        'non_segmented_summarizer_inference': NonSegmentedSummarizerInference,
        'transcript_only_inference': TranscriptOnlyInference,
        'inference': Inference,
        'train': Train,
        'preprocess': Preprocess,
        'evaluate' : Evaluate,
        'extract': Extract
    }

    return pipelines[config.pipeline.name](config)