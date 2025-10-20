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
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPKAN

# Datasets and prediction lengths
dataset=Electricity
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000872897 0.00130044 0.001079003 0.001215337)
batches=(8 8 8 8)
wavelets=(coif5 sym3 sym3 sym3)
levels=(1 1 1 1)
tfactors=(7 7 7 7)
dfactors=(7 7 7 7)
epochs=(20 20 20 20)
dropouts=(0.0 0.0 0.0 0.0)
embedding_dropouts=(0.05 0.1 0.1 0.1)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(128 128 128 128)
patiences=(10 10 10 10)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	log_file="logs/${model_name}/unify_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
