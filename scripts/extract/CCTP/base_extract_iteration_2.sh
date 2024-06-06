#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/data/video/CCTP_iteration_2.yml \
configs/pipeline/extract.yml \
configs/scene_describer/scene_describer.yml \
configs/scene_describer/frames_sampler/base.yml \
configs/reasoner/reasoner.yml \
configs/segmentor/segmentor.yml