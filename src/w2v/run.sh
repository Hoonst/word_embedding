#!/bin/sh

EPOCH=2
BATCH_SIZE=1024

LR=0.1
LR_DECAY_STEP_SIZE=25
LR_DECAY_GAMMA=0.1
WEIGHT_DECAY=0.0001

SEED=42

python main.py\
        --seed=${SEED}\
        --epochs=${EPOCH}\
        --batch-size=${BATCH_SIZE}\
        --lr=${LR}\
        --weight-decay=${WEIGHT_DECAY}\
        --lr-decay-step-size=${LR_DECAY_STEP_SIZE}\
        --lr-decay-gamma=${LR_DECAY_GAMMA}\
        
