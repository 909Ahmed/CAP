import torch
import json
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from model import LoRA
import deepspeed
import logging
import types
import datetime
from functools import partial

def get_phi_model():
    
    bnb_config = {
        "load_in_4bit":True,
        "bnb_4bit_quant_type":"nf4",
        "bnb_4bit_compute_dtype":"bfloat16",
        "bnb_4bit_use_double_quant":True,
    }

    bnb_conf = BitsAndBytesConfig(**bnb_config)    
    
    checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_conf,
        device_map={"": 0},
        use_cache=False,
        attn_implementation='eager', 
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

    for p in model.parameters():
        p.requires_grad = False
    
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05

    assign_lora = partial(LoRA, rank=lora_r, alpha=lora_alpha)
    
    for layer in model.model.layers:
    
        layer.self_attn.o_proj = assign_lora(layer.self_attn.o_proj)
        layer.self_attn.qkv_proj = assign_lora(layer.self_attn.qkv_proj)
        layer.mlp.gate_up_proj = assign_lora(layer.mlp.gate_up_proj)
        layer.mlp.down_proj = assign_lora(layer.mlp.down_proj)
    
    model.lm_head = assign_lora(model.lm_head)
    
    return model

class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.writer = SummaryWriter(args['log_path'])
        
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = 10000
        ds_params['scheduler']['params']['warmup_num_steps'] = 1000
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    def train_model(self, batch, current_step=0, pbar=None):
        
        self.ds_engine.module.train()
        
        loss, mle_acc = self.ds_engine(batch)

        self.writer.add_scalar('loss', loss, current_step)
        self.writer.add_scalar('mle_acc', mle_acc, current_step)
        
        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.update(1)
        
        if self.args['log_path'] and current_step % int(self.args['logging_step']) == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        
        mle_acc *= 100
        return mle_acc
    
def load_model(model, args):
    
    agent = DeepSpeedAgent(model, args)
    return agent