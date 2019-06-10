#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$1 python2.7 train.py --ca --gpu --bs $2
