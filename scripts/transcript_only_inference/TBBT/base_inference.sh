#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/data/utterance_level_transcript/TBBT.yml \
configs/pipeline/transcript_only_inference.yml \
configs/reasoner/reasoner.yml