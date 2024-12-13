max_steps=-1
num_rounds=5
batch_size=4
gradient_accumulation_steps=1
seq_length=512
num_clients=9
sample_clients=9
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5
logging_steps=50
lora_target_modules="q_proj,v_proj"
local_data_dir="/home/tangzichen/ShenNong_TCM_Dataset"       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="michaelwzhu/ShenNong_TCM_Dataset"
dataset_sample=0
# model_name_or_path="/data2/share/Qwen-7b-chat"
model_name_or_path="Qwen/Qwen1.5-0.5B-Chat"
# model_name_or_path="/data2/share/Qwen1.5-7B-chat"
# model_name_or_path="/data2/share/Qwen1.5-0.5B-Chat"
output_dir=./output
use_peft=True
gpu=0
fed_alg="fedavg"
concentration=0.5
partition_type='disease'

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --partition_type $partition_type \
 --concentration $concentration \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --logging_steps $logging_steps \
 --lora_target_modules $lora_target_modules \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \