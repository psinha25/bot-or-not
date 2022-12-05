import os
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from Twibot20Dataset import Twibot20Dataset
from MLPclassifier import MLPclassifier
from utils import create_split_index, eval

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


import matplotlib.pyplot as plt



split = create_split_index()





def fineTune_Roberta(config, checkpoint_dir=None):
    net = MLPclassifier(hidden_dim=config['hidden_dim'], dropout=config['dropout'], input_dim=config['input_dim'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)  

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    train_dataset = Twibot20Dataset('train', split)
    val_dataset = Twibot20Dataset('val', split)
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
   
    
    for epoch in range(100):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            tweet_feature, des_feature, labels = data
            tweet_feature, des_feature, labels = tweet_feature.to(device), des_feature.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(tweet_feature, des_feature)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

       # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                tweet_feature, des_feature, labels = data
                tweet_feature, des_feature, labels = tweet_feature.to(device), des_feature.to(device), labels.to(device)

                outputs = net(tweet_feature, des_feature)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")



def finetune(config, num_samples=10, max_num_epochs=100, gpus_per_trial=2):
    


    scheduler = ASHAScheduler(
        metric="accuracy", #"loss",
        mode="max", #"min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        fineTune_Roberta,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

   


class RobertaTrianer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 epochs=100,
                 input_dim=768,
                 hidden_dim=128,
                 dropout=0.3,
                 activation='relu',
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-6,
                 lr=1e-5,
                 device='cuda:0'):
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        self.device = device
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self):
        train_loader = self.train_loader
        train_loss = []
        valid_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            preds = []
            preds_auc = []
            labels = []
            with tqdm(train_loader) as progress_bar:
                for batch in progress_bar:
                    pred = self.model(batch[0].to(self.device), batch[1].to(self.device))
                    loss = self.loss_func(pred, batch[2].to(self.device))
                    loss_avg += loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())
                    
                    preds.append(pred.argmax(dim=-1).cpu().numpy())
                    preds_auc.append(pred[:, 1].detach().cpu().numpy())
                    labels.append(batch[2].cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            preds_auc = np.concatenate(preds_auc, axis=0)
            labels = np.concatenate(labels, axis=0)
            loss_avg = loss_avg / len(train_loader)   
            print('{' + f'loss={loss_avg.item()}' + '}' + 'eval=', end='')
            eval(preds_auc, preds, labels) 

            train_loss.append(loss_avg.detach().cpu().numpy())    
            valid_loss.append(self.valid().detach().cpu().numpy())
            self.test()
        return train_loss, valid_loss
        
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        loss_avg = 0
        val_loader = self.val_loader
        for batch in val_loader:
            pred = self.model(batch[0].to(self.device), batch[1].to(self.device))
            loss = self.loss_func(pred, batch[2].to(self.device))
            loss_avg += loss
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(batch[2].cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        labels = np.concatenate(labels, axis=0)
        loss_avg = loss_avg / len(val_loader)
        print("Validation Score: ")
        eval(preds_auc, preds, labels)
        return loss_avg
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        test_loader = self.test_loader
        for batch in test_loader:
            pred = self.model(batch[0].to(self.device), batch[1].to(self.device))
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(batch[2].cpu().numpy())
            
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        
        print("Test Score: ")
        eval(preds_auc, preds, labels)
  


parser = argparse.ArgumentParser()
parser.add_argument('--ft', type=bool, default=False)
parser.add_argument('--plot', type=bool, default=False)
args = parser.parse_args() 

if __name__ == '__main__':

    if args.ft == True:
        config = {
        'hidden_dim': tune.sample_from(lambda _: 2**np.random.randint(6, 9)),
        'dropout':tune.loguniform(0.01, 0.4), 
        "lr":tune.loguniform(1e-6, 1e-4), 
        "weight_decay": tune.loguniform(1e-7, 1e-5),
        "input_dim": 1024
        } 

        finetune(config, num_samples=100, max_num_epochs=100, gpus_per_trial=1)
    else:
        train_dataset = Twibot20Dataset('train', split)
        val_dataset = Twibot20Dataset('val', split)
        test_dataset = Twibot20Dataset('test', split)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        trainer = RobertaTrianer(train_loader, val_loader, test_loader, epochs=50, input_dim=1024, hidden_dim=256, dropout=0.224, lr=9.86e-05, weight_decay=7.395e-07)
        train_loss, valid_loss = trainer.train()

        def argmin(iterable):
            return min(enumerate(iterable), key=lambda x: x[1])[0]
        min_valid_loss_arg = argmin(valid_loss)
        min_valid_loss = valid_loss[min_valid_loss_arg]

        print(f"Min loss epoch: {min_valid_loss_arg} Min loss: {min_valid_loss}")
        
        if args.plot == True:
            plt.plot(train_loss)
            plt.plot(valid_loss)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f"Min loss: {min_valid_loss:.3f} at epoch: {min_valid_loss_arg}")


            plt.savefig("./plot/loss_plot")


    

    