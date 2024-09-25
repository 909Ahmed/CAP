import torch 
import torch.nn as nn
import requests
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from PIL import Image

class DaVaDataset(Dataset):
        
    def __init__(self, data, preprocessor):
        self.data = data
        self.processor = preprocessor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        text = self.data[idx][1]
        image_url = self.data[idx][2]
        x = requests.get(image_url, stream=True)
        try:
            image = Image.open(x.raw)
            image = self.processor(image)
        except:
            image = torch.zeros((3, 224, 224))
            print("shitty image")

        return {"images" : image, "captions" : text}

def collate_fn(batch):
    return {
        'image': torch.stack([x['images'] for x in batch]),
        'captions': [x['captions'] for x in batch]      
    }