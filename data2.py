import torch 
from io import BytesIO
from torch.utils.data import Dataset
from PIL import Image

class DaVaDataset(Dataset):
    def __init__(self, data, preprocessor):
        self.data = data
        self.processor = preprocessor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        try:
            
            with torch.no_grad():
                image = Image.open(BytesIO(self.data[idx]['image']))
                image = self.processor(image)
            
        except:
            image = torch.zeros((3, 224, 224))
            print("shitty image")

        conversation = self.data[idx]['conversations']
        
        return {"image" : image, "conversation" : conversation}

def collate_fn(batch):
    return {
        'images': torch.stack([x['image'] for x in batch]),
        "conversations" : [x['conversation'] for x in batch]
    }