from src.utils import Config
from .chatgpt_evaluator import ChatGPTEvaluator
from .chatgpt_zero_shot_evaluator import ChatGPTZeroShotEvaluator

def get_evaluator(config: Config):
    
    evaluators = {
        'gpt-4': ChatGPTEvaluator,
        'zero-shot-gpt-4': ChatGPTZeroShotEvaluator,
        'zero-shot-gpt-35': ChatGPTZeroShotEvaluator,
        'zero-shot-gpt-4-i2': ChatGPTZeroShotEvaluator,
        'zero-shot-gpt-35-i2': ChatGPTZeroShotEvaluator,
        'zero-shot-gpt-4-i1-wo-tracking': ChatGPTZeroShotEvaluator,
        'zero-shot-gpt-4-i1-fewshot': ChatGPTZeroShotEvaluator
    }

    return evaluators[config.evaluator.name](config.evaluator)