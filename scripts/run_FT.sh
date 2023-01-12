#!/bin/bash

# Example run: bash run_FT.sh [TASK] [EX_NAME_SUFFIX]

glue_low=(MRPC RTE STS-B CoLA)
glue_high=(MNLI QQP QNLI SST-2)

proj_dir=CoFI_runs

code_dir=.


# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=bert-base-uncased

# logging & saving
logging_steps=100
save_steps=0
if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
fi

# train parameters
max_seq_length=128
# batch_size=32

# epochs=5

# seed
seed=57

# output directory
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix}


pretrained_pruned_model=${4}
learning_rate=$3
scheduler_type=none
output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
epochs=50 #20
batch_size=64


mkdir -p $output_dir
pruning_type=None


python3 $code_dir/run_glue_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --pretrained_pruned_model ${pretrained_pruned_model} \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} 2>&1 | tee $output_dir/all_log.txt

