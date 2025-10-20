#!/bin/bash

# 创建logs目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# 创建WPKAN logs目录
if [ ! -d "./logs/WPKan-Mixer" ]; then
    mkdir ./logs/WPKan-Mixer
fi

# Create the hyperParameterSearchOutput directory if it doesn't exist
if [ ! -d "./hyperParameterSearchOutput" ]; then
    mkdir ./hyperParameterSearchOutput
fi

# 设置要使用的GPU
export CUDA_VISIBLE_DEVICES=0

# unified-TimeMixer setting
seq_len=96
batch_size=32
epochs=20

# 模型名称
model_name=WPKAN
task_name=long_term_forecast
loss_name=smoothL1
patience=10

# 数据集和预测步长
dataset=Electricity
pred_len=720
trial_num=200

log_file="logs/WPKan-Mixer/full_hyperParameter_Search_result_${dataset}_${pred_len}.log"
echo "Starting hyperparameter optimization for pred_len: ${pred_len}"

python -u run_LTF.py \
    --task_name $task_name \
    --model $model_name \
    --data $dataset \
    --pred_len $pred_len \
    --features M \
    --use_hyperParam_optim \
    --itr 1 \
    --use_gpu True \
    --gpu 0 \
    --use_amp \
    --loss $loss_name \
    --seed 42 \
    --n_jobs 1 \
    --optuna_trial_num $trial_num \
    --optuna_seq_len $seq_len \
    --optuna_lr 0.00001 0.01 \
    --optuna_batch $batch_size \
    --optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif1 coif4 coif5 bior3.1 \
    --optuna_tfactor 3 5 7 \
    --optuna_dfactor 3 5 7 8 \
    --optuna_epochs $epochs \
    --optuna_dropout 0.0 0.05 0.1 0.2 0.3 0.4 0.5 \
    --optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
    --optuna_patch_len 12 16 \
    --optuna_stride 4 8 \
    --optuna_dmodel 128 256 \
    --optuna_weight_decay 0.0 \
    --optuna_patience $patience \
    --optuna_level 1 2 3 \
    --optuna_lradj type1 type2 type3 > $log_file