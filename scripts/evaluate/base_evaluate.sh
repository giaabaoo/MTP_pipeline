#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/pipeline/evaluate.yml \
configs/evaluator/evaluator.yml