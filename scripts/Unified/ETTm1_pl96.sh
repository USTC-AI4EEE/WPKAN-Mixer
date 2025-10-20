#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Create WPKAN logs directory if it doesn't exist
if [ ! -d "./logs/WPKan-Mixer" ]; then
    mkdir ./logs/WPKan-Mixer
fi

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=1

# Model name
model_name=WPKAN

# Datasets and prediction lengths
dataset=ETTm1
seq_lens=96
pred_lens=96
learning_rates=0.000998863786869981
batches=64
wavelets=db3
levels=2
tfactors=3
dfactors=8
epochs=10
dropouts=0.2
embedding_dropouts=0.05
patch_lens=16
strides=4
lradjs=type2
d_models=256
patiences=5
weight_decay=0.0


# Loop over datasets and prediction lengths
log_file="logs/WPKan-Mixer/full_hyperUnified_result_${dataset}_${pred_lens[$i]}.log"
python -u run_LTF.py \
	--model $model_name \
	--task_name long_term_forecast \
	--data $dataset \
	--seq_len $seq_lens \
	--pred_len $pred_lens \
	--d_model $d_models \
	--tfactor $tfactors \
	--dfactor $dfactors \
	--wavelet $wavelets \
	--level $levels \
	--patch_len $patch_lens \
	--stride $strides \
	--batch_size $batches \
	--learning_rate $learning_rates \
	--lradj $lradjs \
	--dropout $dropouts \
	--embedding_dropout $embedding_dropouts \
	--patience $patiences \
	--train_epochs $epochs \
    --weight_decay $weight_decay \
	--use_amp > $log_file

