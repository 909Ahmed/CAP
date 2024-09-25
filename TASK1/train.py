import torch
from tqdm import tqdm
import pickle
from data import DaVaDataset, collate_fn
from model import DaVa
from utils import *
import clip

def train_model():

    clipped, clippreprocess = clip.load("ViT-B/32", device='cuda')
    train_data = pickle.load(open('./task1/CAP/TASK1/dataset.pkl', 'rb'))
    train_dataset = DaVaDataset(train_data, clippreprocess)
    batch_size = 2
    train_iter = torch.utils.data.DataLoader(train_dataset, 
                                            num_workers=4,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            collate_fn=collate_fn)
    
    args = {
        
        "epochs" : 1,
        "world_size" : 1,
        "save_path_phi" : "./task1/CAP/TASK1/save_phi.pkl",
        "save_path_proj" : "./task1/CAP/TASK1/save_proj.pkl",
        "log_path":"./task1/CAP/TASK1/log.txt",
        "stage" : 1,
        "total_steps" : "1000",
        "warmup_rate": "0.1",
        "max_length": "512",
        "ds_config_path" : "./task1/CAP/TASK1/dataset.json",
        "seed" : "13",
        "max_length": "512",
        "logging_step" : "5"      
    }
    
    phi_model = get_phi_model()
    model = DaVa(phi_model=phi_model)
    model.freeze()
    
    # check out this shit?
    train_num = train_data.__len__()
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
            if current_step % 2000 == 0:
                with open(args['save_path_phi'], 'wb') as phi_file:
                    pickle.dump(agent.model.phi.state_dict(), phi_file)
                with open(args['save_path_proj'], 'wb') as proj_file:
                    pickle.dump(agent.model.proj.state_dict(), proj_file)

    with open(args['save_path_phi'], 'wb') as phi_file:
            pickle.dump(agent.model.phi.state_dict(), phi_file)
    with open(args['save_path_proj'], 'wb') as proj_file:
        pickle.dump(agent.model.proj.state_dict(), proj_file)