#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
export CUDA_VISIBLE_DEVICES="0,1,2"


model=HSLLM
device=$1
prompt=$2
seq_len_values=($3)
pred_len_values=($4)
gpt_layers_values=($5)

for seq_len in "${seq_len_values[@]}"
do
for pred_len in "${pred_len_values[@]}"
do
for gpt_layers in "${gpt_layers_values[@]}"
do
python test.py \
    --device $device \
    --root_path "/home/lc/TimeSeries/datasets/" \
    --data_path 'TSF_kiln.csv'\
    --model_id $model'_'$prompt'_gl'$gpt_layers'_sl'$seq_len'_pl'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 4 \
    --c_out 4 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layers $gpt_layers \
    --itr 2 \
    --model $model \
    --is_gpt 1 \
    --prompt $prompt 
    
done
done
done