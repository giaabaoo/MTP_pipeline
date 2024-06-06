#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/data/video/CCTP_balance_set.yml \
configs/pipeline/extract.yml \
configs/scene_describer/gpt4v.yml \
configs/scene_describer/frames_sampler/base.yml \
configs/reasoner/reasoner.yml \
configs/segmentor/segmentor.yml