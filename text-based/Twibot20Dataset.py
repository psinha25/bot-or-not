
from torch.utils.data import Dataset
from pathlib import Path
import torch

class Twibot20Dataset(Dataset):
    
    def __init__(self, name, split):
        path = Path('/data/hchangac/ML/data')
        
        tweets_tensor = torch.load(path / 'tweets_tensor_roberta_large.pt')
        des_tensor = torch.load(path / 'des_tensors_roberta_large.pt')
        label = torch.load(path / 'label_tensors.pt')
        
        
        if name == 'train':
            self.tweet_feature = tweets_tensor[split[0]]
            self.des_feature = des_tensor[split[0]]
            self.label = label[split[0]]
            self.length = len(self.tweet_feature)
        elif name == 'val':
            self.tweet_feature = tweets_tensor[split[1]]
            self.des_feature = des_tensor[split[1]]
            self.label = label[split[1]]
            self.length = len(self.tweet_feature)
        elif name == 'train_p_val':
            all_train = split[0] + split[1]
            self.tweet_feature = tweets_tensor[all_train]
            self.des_feature = des_tensor[all_train]
            self.label = label[all_train]
            self.length = len(self.tweet_feature)           
        else:
            self.tweet_feature = tweets_tensor[split[2]]
            self.des_feature = des_tensor[split[2]]
            self.label = label[split[2]]
            self.length = len(self.tweet_feature)
        """
        batch_size here is useless
        """
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.tweet_feature[index], self.des_feature[index], self.label[index]
