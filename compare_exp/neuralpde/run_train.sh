#!/bin/bash
echo "All tasks started."

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data burgers --method rk4 --master_port 20505 &

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data darcy --method rk4 --master_port 20506 &

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py --data ns --method rk4 --master_port 20507 &

wait
echo "All tasks completed."
