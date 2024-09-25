import torch
import json
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import logging
import types
import datetime

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
    
    return model

class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.writer = SummaryWriter(args['log_path'])
        
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(int(self.args['total_steps']) * float(self.args['warmup_rate'])))
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()

        loss, mle_acc, mse_loss = self.ds_engine(batch)

        self.writer.add_scalar('loss', loss, current_step)
        self.writer.add_scalar('mle_acc', mle_acc, current_step)
        
        self.ds_engine.backward(loss)
        self.ds_engine.step()
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}; mse_loss: {round(mse_loss[0].item(), 4)} ')
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['log_path'] and current_step % int(self.args['logging_step']) == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        mle_acc *= 100
        return mle_acc
    
def load_model(model, args):
    
    agent = DeepSpeedAgent(model, args)
    return agent