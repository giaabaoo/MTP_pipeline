#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/pipeline/preprocess.yml \
configs/preprocessor/preprocessor_balance_set.yml