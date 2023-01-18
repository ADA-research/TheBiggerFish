import torch
from torch.utils.data import Dataset 

class DataFrameTorch(Dataset):
    def __init__(self,x, y, indexreturn = False):
        self.x_train = torch.tensor(x.values, dtype = torch.float)
        self.y_train = torch.reshape(torch.tensor(y.values, dtype = torch.float),(y.shape[0],1))
        self.indexreturn = indexreturn
    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        if self.indexreturn:
            return self.x_train[idx],self.y_train[idx],idx
        else: 
            return self.x_train[idx],self.y_train[idx]


