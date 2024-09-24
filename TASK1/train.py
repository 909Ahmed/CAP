import torch #type:ignore
from tqdm import tqdm #type:ignore
import pickle
from data import DaVaDataset, collate_fn
from model import DaVa
from utils import *

def train():

    clip, clippreprocess = clip.load("ViT-B/32", device='cuda')
    train_data = pickle.load(open('train_data.pkl', 'rb'))
    train_dataset = DaVaDataset(train_data, clippreprocess)
    batch_size = 2
    train_iter = torch.utils.dataloader = torch.utils.DataLoader(train_dataset, 
                                                                 num_workers=4,
                                                                 batch_size=batch_size, 
                                                                 suffle=True,
                                                                 collate_fn=collate_fn)
    
    args = {
        
        "epochs" : 1,
        "world_size" : 1,
        "save_path_phi" : "save_phi.pkl",
        "save_path_" : "",
        "log_path" : "log.txt",
        "stage" : 1,
        "total_steps" : "1000",
        "warmup_rate": "0.1",
        "max_length": "512",
        "ds_config_path" : "dataset.json"
                
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
                pickle.dump(args['save_path_phi'], )
                
    pickle.dump()