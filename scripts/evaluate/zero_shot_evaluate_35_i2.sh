#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/pipeline/evaluate.yml \
configs/evaluator/zero_shot_evaluator_35_i2.yml