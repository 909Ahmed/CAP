import torch
from tqdm import tqdm
import pickle
from data import DaVaDataset, collate_fn
from model import DaVa
from utils import *
import clip

def train_model():

    clipped, clippreprocess = clip.load("ViT-B/32", device='cuda')
    train_data = pickle.load(open('./dataset.pkl', 'rb'))
    train_dataset = DaVaDataset(train_data, clippreprocess)
    batch_size = 16
    train_iter = torch.utils.data.DataLoader(train_dataset, 
                                            num_workers=8,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            collate_fn=collate_fn)
    
    args = {
        
        "epochs" : 1,
        "world_size" : 1,
        "save_path_proj" : "./states/",
        "log_path":"./log.txt",
        "stage" : 1,
        "total_steps" : "1000",
        "warmup_rate": "0.1",
        "max_length": "512",
        "ds_config_path" : "./dataset.json",
        "seed" : "13",
        "max_length": "512",
        "logging_step" : "25"      
    }
    
    phi_model = get_phi_model()
    model = DaVa(phi_model=phi_model)
    model.stage = 2
    
    # check out this shit?
    train_num = 10000
    length = args['epochs'] * train_num // args['world_size']
    total_steps = args['epochs'] * train_num // batch_size
    args['total_steps'] = total_steps
    agent = load_model(model, args)

    pbar = tqdm(total=length)  # maximum total number
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
                torch.save(agent.model.proj.state_dict(), str(args['save_path_proj'] + f'{current_step}.pkl'))

    torch.save(agent.model.proj.state_dict(), str(args['save_path_proj'] + '10000.pkl'))