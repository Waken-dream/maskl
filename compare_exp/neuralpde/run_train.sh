#!/bin/bash
echo "All tasks started."

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data burgers --master_port 20502 &

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data darcy --master_port 20503 &

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data ns --master_port 20504 &

wait
echo "All tasks completed."