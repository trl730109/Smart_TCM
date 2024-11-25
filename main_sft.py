import copy
import os
import math
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, TrainingArguments, Trainer
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from federated_learning.split_dataset import *
from config import get_config, save_config, get_model_config, get_training_args
        
# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

train_test_split = dataset.train_test_split(test_size=0.1, seed=2023)

# Access the training and test sets
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print(f'Train dataset size: {len(train_dataset)}')
print(f'Train dataset size: {len(test_dataset)}')

# ===== Split the dataset into clients =====
# local_datasets = split_dataset(fed_args, script_args, train_dataset)
local_datasets = partition_dataset_with_quantity_skew(fed_args, train_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(f'Partitioned dataset size: {sample_num_list}')

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_safetensors=True
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
test_loss_list = []
test_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True)

test_trainer = SFTTrainer(model=model, 
                tokenizer = tokenizer, 
                args=test_args, 
                compute_metrics=None, 
                eval_dataset = test_dataset,
                formatting_func=formatting_prompts_func,
                )
eval_results = test_trainer.evaluate()
print(eval_results)
test_loss_list.append(eval_results['eval_loss'])
print(f'Before training test loss: {eval_results}')

train_loss_step = {i:[] for i in range(fed_args.num_clients)}
for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        # global_dict_copy = copy.deepcopy(global_dict)
        # initial_model_weights = copy.deepcopy(get_peft_model_state_dict(model))
        # print("Initial model LoRA weights captured.")

        set_peft_model_state_dict(model, global_dict)
        # def compare_state_dicts(dict1, dict2):
        #     for key in dict1:
        #         if not torch.equal(dict1[key], dict2[key]):
        #             return False
        #     return True

        # if compare_state_dicts(initial_model_weights, global_dict):
        #     print("The base model already load the lastest global lora weights.")
        # else:
        #     print("Load global LoRA unsuccessfully.")
        # sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        sub_dataset = local_datasets[client]
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        loss_epoch_list = []
        for elem in trainer.state.log_history:
            if 'loss' in elem.keys():
                loss_epoch_list.append(elem['loss'])
        train_loss_step[client].append(loss_epoch_list)
        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
        set_peft_model_state_dict(model, global_dict)

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    
    test_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True)

    test_trainer = SFTTrainer(model=model, 
                    tokenizer = tokenizer, 
                    args=test_args, 
                    compute_metrics=None, 
                    eval_dataset = test_dataset,
                    formatting_func=formatting_prompts_func,
                    )
    eval_results = test_trainer.evaluate()
    print(eval_results)
    test_loss_list.append(eval_results['eval_loss'])

json_string = json.dumps(test_loss_list)
json_path = os.path.join(script_args.output_dir , 'test_losses.json')
with open(json_path, 'w') as file:
    file.write(json_string)
    
file_path = os.path.join(script_args.output_dir , 'client_train_loss.json')
with open(file_path, 'w') as json_file:
    json.dump(train_loss_step, json_file, indent=4)
    
print(f'Data successfully saved to {file_path}')
    
# print(f'Test on the test datasets:')
# print("Number of samples in the test dataset:", len(test_dataset))
# test_args = TrainingArguments(
#         output_dir='./results',
#         per_device_eval_batch_size=8,
#         do_train=False,
#         do_eval=True)

# trainer = SFTTrainer(model=model, 
#                   tokenizer = tokenizer, 
#                   args=test_args, 
#                   compute_metrics=None, 
#                   eval_dataset = test_dataset,
#                   formatting_func=formatting_prompts_func,
#                   )
# eval_results = trainer.evaluate()
# print(eval_results)
