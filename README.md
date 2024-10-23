# Mixture-of-Skills
Code for the EMNLP2024 main paper ["Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models"](https://arxiv.org/abs/2406.08811)

## Setup the environment

```
conda create -n mixture_of_skills python=3.10
conda activate mixture_of_skills

git clone https://github.com/minghao-wu/mixture_of_skills.git
cd mixture_of_skills
pip install -r requirements.txt
```

## Training

For demonstration purposes, we provide several datasets at `mixture_of_skills/datasets`.

```
model=$1 # e.g. meta-llama/Llama-3.1-8B
reward_type=$2 # "cossim" or "diff"
update_steps=$3 # interval for updating the scorer network e.g. 200
temperature=$4 # initial sampling probability, e.g. 1, inf
use_ema=$5 # use ema or not "yes" or "no"
ema_alpha=$6 # beta in equation 11. e.g. 0.9
output_dir=$7 # directory saving checkpoint
dataset_dir=$8 # dataset directory. e.g. mixture_of_skills/datasets
data_names=$9 # the names of subset, delimited by "_". e.g. "general_math_medical_p3"


ds_config=deepspeed_config_bf16_stage2.json
run_script=run_clm.py

mkdir -p $output_dir

deepspeed --num_nodes 1 --num_gpus 2 \
    $run_script \
    --deepspeed $ds_config \
    --model_name_or_path $model \
    --dataset_dir $dataset_dir \
    --dataset_names $data_names \
    --dataset_init_temperature $temperature \
    --trainer_type "dynamic" \
    --sampling_prob_update_steps $update_steps \
    --reward_type $reward_type \
    --use_ema $use_ema \
    --ema_alpha $ema_alpha \
    --output_dir $output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --seed 42 \
    --do_train \
    --bf16 \
    --report_to "wandb" \
    --run_name $(basename $output_dir) \
    --gradient_checkpointing \

```

## Citation

```
@article{wu2024mixture,
  title={Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models},
  author={Wu, Minghao and Vu, Thuy-Trang and Qu, Lizhen and Haffari, Gholamreza},
  journal={arXiv preprint arXiv:2406.08811},
  year={2024}
}
```