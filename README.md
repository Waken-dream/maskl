# Mask Learning For Time Dependent PDEs

## Setup
```bash
conda create -n mask python=3.11
conda activate mask
conda install --file requirements.txt
```

## Run

## Experiment

证明在数据有缺失时，原有网络丧失预测能力

Exp1: 有缺失数据 + 别人网络（FNO，iTransformer, NeuralPDE）预测结果

Exp2: 恢复网络恢复出的数据 + 别人网络预测结果

先测试现有网络在有缺失数据集上的预测能力，再测试现有网络在恢复网络恢复出的数据上的预测能力


# Questions
为什么要使用自己的预测网络？恢复网络是否可以有通用性