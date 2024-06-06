#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/pipeline/train.yml \
configs/preprocessor/preprocessor.yml \
configs/trainer/trainer.yml