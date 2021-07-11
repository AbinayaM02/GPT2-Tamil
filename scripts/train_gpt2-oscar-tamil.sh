#!/usr/bin/env bash
python ../src/run_clm_flax.py \
    --output_dir="../tmp/${MODEL_DIR}" \
    --model_type="gpt2" \
    --config_name="${MODEL_DIR}" \
    --tokenizer_name="${MODEL_DIR}" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_ta" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="3e-5" \
    --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --report_to wandb \
    --run_name trial \
    #--push_to_hub
    2>&1 | tee run.log
