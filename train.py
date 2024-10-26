import torch
from tqdm import tqdm
import pickle
from data2 import DaVaDataset, collate_fn
from model import DaVa
from utils import *
import clip

def train_model():

    clippreprocess = clip.load("ViT-B/32", device='cuda')[1]
    train_data = pickle.load(open('/teamspace/studios/this_studio/NTASK/images.pkl', 'rb'))
    train_dataset = DaVaDataset(train_data, clippreprocess)
    batch_size = 8
    train_iter = torch.utils.data.DataLoader(train_dataset, 
                                            num_workers=2,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            collate_fn=collate_fn)
    
    args = {
        
        "epochs" : 2,
        "world_size" : 1,
        "save_path_proj" : "/teamspace/studios/this_studio/NTASK/final_states",
        "save_path_phi" : "/teamspace/studios/this_studio/NTASK/final_statues",
        "log_path":"./log.txt",
        "stage" : 1,
        "total_steps" : "10000",
        "warmup_rate": "0.1",
        "max_length": "512",
        "ds_config_path" : "/teamspace/studios/this_studio/NTASK/dataset.json",
        "seed" : "42",
        "max_length": "512",
        "logging_step" : "10"
    }
    
    phi_model = get_phi_model()
    model = DaVa(phi_model=phi_model)
    for p in model.proj.clip.parameters():
        p.requires_grad = False
    
    # model.apply_lora_to_proj(rank=64, alpha=128)
    model.stage = 2
    
    # check out this shit?
    train_num = 50000
    length = args['epochs'] * train_num // args['world_size']
    total_steps = args['epochs'] * train_num // batch_size
    args['total_steps'] = total_steps
    agent = load_model(model, args)

    pbar = tqdm(total=total_steps // 2)  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        
        for batch in train_iter:
            
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )
            
            current_step += 1
            if current_step % 1000 == 0:

                torch.save(agent.model.phi.state_dict(), str(args['save_path_phi'] + '2' + f'{current_step}.pkl'))
                torch.save(agent.model.proj.state_dict(), str(args['save_path_proj'] + '2' + f'{current_step}.pkl'))

    torch.save(agent.model.proj.state_dict(), str(args['save_path_proj'] + '420.pkl'))