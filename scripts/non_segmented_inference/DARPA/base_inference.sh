#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/data/utterance_level_video/DARPA.yml \
configs/pipeline/non_segmented_inference.yml \
configs/scene_describer/scene_describer.yml \
configs/scene_describer/frames_sampler/base.yml \
configs/reasoner/reasoner.yml \
configs/segmentor/segmentor.yml