import os
import numpy as np
import pandas as pd
import torch
import argparse

from sklearn.model_selection import KFold
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from Twibot20Dataset import Twibot20Dataset
from MLPclassifier import MLPclassifier
from utils import create_split_index



split = create_split_index()

class RobertaOOFTrianer:
    def __init__(self,
                 train_data,
                 test_data,
                 epochs=100,
                 input_dim=1024,
                 hidden_dim=128,
                 dropout=0.3,
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-6,
                 lr=1e-5,
                 device='cuda:0'):
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.weight_decay = weight_decay
        self.lr = lr
        self.optimizer = optimizer
        
        self.train_data = train_data
        self.test_data = test_data
        self.loss_func = nn.CrossEntropyLoss()


    def oof_pred(self, k_folds):

        kfold=KFold(n_splits=k_folds,shuffle=True)

        y_oof = np.zeros(len(self.train_data))
        y_test = np.zeros(len(self.test_data))

        test_loader = DataLoader(self.test_data, batch_size=2, shuffle=False)
        
        for fold,(train_idx,val_idx) in enumerate(kfold.split(self.train_data)):
            print('------------fold no---------{}----------------------'.format(fold))
            train_set = torch.utils.data.Subset(self.train_data,train_idx)
            val_set = torch.utils.data.Subset(self.train_data,val_idx)

            train_loader = torch.utils.data.DataLoader(
                                train_set,
                                shuffle=True, 
                                batch_size=32, num_workers=8)
            val_loader = torch.utils.data.DataLoader(
                                val_set,
                                shuffle=False, 
                                batch_size=32, num_workers=8)

            model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)
            model.to(self.device)

            optimizer = self.optimizer(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.train(model, optimizer, train_loader)

            preds = self.test(model, val_loader)
            preds_test = self.test(model, test_loader)


            y_oof[val_idx] = y_oof[val_idx] + preds
            y_test = y_test + preds_test / k_folds


        return y_oof, y_test




    def train(self, model, optimizer, train_loader):
        for epoch in range(self.epochs):
            model.train()
            with tqdm(train_loader) as progress_bar:
                for batch in progress_bar:
                    pred = model(batch[0].to(self.device), batch[1].to(self.device))
                    loss = self.loss_func(pred, batch[2].to(self.device))

                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())
   

    @torch.no_grad()
    def test(self, model, data_loader):
        model.eval()
        preds_auc = []
        for batch in data_loader:
            pred = model(batch[0].to(self.device), batch[1].to(self.device))
            preds_auc.append(pred[:,1].detach().cpu().numpy())


        preds_auc = np.concatenate(preds_auc, axis=0)

        return preds_auc


if __name__ == '__main__':

    train_dataset = Twibot20Dataset('train_p_val', split)
    test_dataset = Twibot20Dataset('test', split)
    

    trainer = RobertaOOFTrianer(train_dataset, test_dataset, epochs=17, input_dim=1024, hidden_dim=256, dropout=0.224, lr=9.86e-05, weight_decay=7.395e-07)
    y_oof, y_test = trainer.oof_pred(5)
    np.save("../data/y_oof", y_oof)
    np.save("../data/y_test", y_test)