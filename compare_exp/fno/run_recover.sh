#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --data burgers --action recover --epoch 30000 &

CUDA_VISIBLE_DEVICES=2 python main.py --data darcy --action recover --epoch 30000 &

CUDA_VISIBLE_DEVICES=3 python main.py --data ns --action recover --epoch 30000 &

CUDA_VISIBLE_DEVICES=4 python main.py --data burgers --action recover --epoch 30000 --sim 0 &

CUDA_VISIBLE_DEVICES=5 python main.py --data darcy --action recover --epoch 30000 --sim 0 &

CUDA_VISIBLE_DEVICES=6 python main.py --data ns --action recover --epoch 30000 --sim 0 &

wait
echo "All tasks completed."