import pickle 

with open('images.pkl', 'rb') as file:
    data = pickle.load(file)
    
print(len(data))