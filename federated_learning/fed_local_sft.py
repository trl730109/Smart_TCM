import torch
import copy
import math
from trl import SFTTrainer
from transformers import TrainerCallback, Trainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            # logging_dir='./logs',
            # logging_strategy="steps",
            # logging_steps=50,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer


class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

class LLMTrainer:
    def __init__(self, net, test_dl, device):
        self.net = net
        self.test_dl = test_dl
        self.device = device
        
    def to_device(self, batch):
        """Move batch of data into device memory."""
        if type(batch) == dict: 
            keys = list(batch.keys())
            device_batch = {
                k: v.to(device=self.device, dtype=torch.long if k == "input_ids" else None, non_blocking=True)
                for k, v in batch.items()
                if k in keys  # Add more keywords here if needed
            }
        elif type(batch) == list:
            device_batch = [
                v.to(device=self.device, non_blocking=True)
                for v in batch
            ]
        else:
            raise RuntimeError
        return device_batch

    def test(self):
            self.net.eval()
            test_loss = 0
            total = 0
            total_iters = 0

            with torch.no_grad():
                for step, batch in enumerate(self.test_dl):
                    device_batch = self.to_device(batch)
                    outputs = self.net(**device_batch)
                    loss = outputs.loss
                    batch_size = device_batch["labels"].size(0)
                    test_loss += loss.data.item()
                    total += device_batch["labels"].size(0)
                    total_iters += 1

            test_loss /= total_iters
            test_ppl = math.exp(test_loss)
            # print('Epoch %d, lr: %f, val loss: %f, val ppl: %f' % (epoch, self.lr, test_loss, test_ppl))
            self.net.train()
            return test_ppl, test_loss

class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)