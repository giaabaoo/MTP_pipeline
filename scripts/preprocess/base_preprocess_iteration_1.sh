#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/pipeline/preprocess.yml \
configs/preprocessor/preprocessor_iteration_1.yml