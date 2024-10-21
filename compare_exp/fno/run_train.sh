#!/bin/bash
echo "All tasks started."

python main.py --data burgers --action train &

python main.py --data darcy --action train &

python main.py --data ns --action train &

wait
echo "All tasks completed."