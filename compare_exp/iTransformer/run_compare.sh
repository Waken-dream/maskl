#!/bin/bash

model_name=iTransformer
echo "All tasks started."

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py \
   --data burgers \
   --master_port 20510 \
   --use_multi_gpu \
   --resolution 1024 \
   --enc_in 1024 \
   --seq_len 1 \
   --pred_len 1 &

torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py \
   --data darcy \
   --batch_size 8 \
   --master_port 20511 \
   --use_multi_gpu \
   --resolution 49 \
   --enc_in 2401 \
   --seq_len 1 \
   --pred_len 1 &
   
torchrun --standalone --nnodes 1 --nproc_per_node 8 train.py \
   --data ns \
   --batch_size 8 \
   --master_port 20512 \
   --use_multi_gpu \
   --resolution 64 \
   --enc_in 4096 \
   --seq_len 10 \
   --pred_len 10 &

wait
echo "All tasks completed."