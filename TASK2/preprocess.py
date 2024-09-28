import json
import requests
from tqdm import tqdm
import lzma
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

data_path = './llava_instruct_150k.json'

with open(data_path, 'r') as file:
    dataset = json.load(file)

def process_data(data):
    answer = {}
    image_url = 'http://images.cocodataset.org/train2017/' + str(data['image'])

    try:
        x = requests.get(image_url, stream=True, timeout=2)
        answer['image'] = x.content
        answer['conversations'] = data['conversations']
        
    except Exception as e:
        print("Error fetching image", e)
    
    return answer

alls = []

with ThreadPoolExecutor(max_workers=100) as executor:

    futures = [executor.submit(process_data, data) for data in dataset]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
        result = future.result()
        if result:
            alls.append(result)
            
        if len(alls) % 100 == 0:
            print(len(alls))

        if len(alls) == 10000:
            pickle.dump(alls, lzma.open('images.pkl.xz', 'wb'))