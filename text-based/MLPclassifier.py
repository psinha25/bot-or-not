import torch
from torch import nn



class MLPclassifier(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=128,
                 dropout=0.3):
        super(MLPclassifier, self).__init__()
        self.dropout = dropout
        
        self.pre_model1 = nn.Linear(input_dim, input_dim // 2)
        self.pre_model2 = nn.Linear(input_dim, input_dim // 2)
        
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self,tweet_feature, des_feature):
        pre1 = self.pre_model1(tweet_feature)
        pre2 = self.pre_model2(des_feature)
        x = torch.cat((pre1,pre2), dim=1)
        x = self.linear_relu_tweet(x)
        # x = self.linear_relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x  