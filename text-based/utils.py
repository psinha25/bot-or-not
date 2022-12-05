import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

def eval(preds_auc, preds, labels):
    print("ACC:{}".format(accuracy_score(labels, preds)), end=",")
    print("F1:{}".format(f1_score(labels, preds)), end=",")
    print("ROC:{}".format(roc_auc_score(labels, preds_auc)))
    print("precision_score:{}".format(precision_score(labels, preds)), end=",")
    print("recall_score:{}".format(recall_score(labels, preds)))



def generate_tweet_embedding(model_name, max_tweet=10):
    data = pd.read_json('../data/all_data.json')
    users_tweets = data['tweet']

    tweets_tensor_list = []

    feature_extractor = pipeline('feature-extraction', model=model_name,tokenizer=model_name, device=0, padding=True, truncation=True, max_length=50)

    
    for i in tqdm(range(len(data))):
        user_tweets_tensor = []
        try:
            count = 0
            for n, tweet in enumerate(users_tweets[i]):
                if count == max_tweet: # We only consider #max_tweet tweets
                    break
                tweet_tensor = torch.tensor(feature_extractor(tweet)).squeeze(0)
                tweet_tensor = torch.mean(tweet_tensor, dim=0)
                user_tweets_tensor.append(tweet_tensor)
                count += 1

            user_tweets_tensor = torch.mean(torch.stack(user_tweets_tensor), dim=0)
        except:
            user_tweets_tensor = torch.zeros(1024)
        
        tweets_tensor_list.append(user_tweets_tensor)


    path3 = Path('/data/hchangac/ML/data')
    tweets_tensor = torch.stack(tweets_tensor_list)
    torch.save(tweets_tensor, path3 / f'tweets_tensor_{model_name}.pt')




def generate_des_embedding(model_name):
    data = pd.read_json('../data/all_data.json')
    des_tensors = []
    feature_extractor = pipeline('feature-extraction', model=model_name,tokenizer=model_name, device=0, padding=True, truncation=True, max_length=50)


    for i in tqdm(range(len(data))):
        try:
            des = data['profile'][i]['description']
            des_tensor = torch.tensor(feature_extractor(des)).squeeze(0)
            des_tensor = torch.mean(des_tensor, dim=0)
        except:
            des_tensor = torch.zeros(768)
        
        des_tensors.append(des_tensor)


    path3 = Path('/data/hchangac/ML/data')
    des_tensors = torch.stack(des_tensors)
    torch.save(des_tensors, path3 / f'des_tensors_{model_name}.pt')


def generate_label_list():
    data = pd.read_json('../data/all_data.json')
    label_tensor = torch.tensor(data.label)
    path3 = Path('/data/hchangac/ML/data')
    torch.save(label_tensor, path3 / 'label_tensors.pt')

def combine_data():
    df_train=pd.read_json('../data/train.json')
    df_dev=pd.read_json('../data/dev.json')
    df_test=pd.read_json('../data/test.json')

    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)

    df_all.to_json('../data/all_data.json')


def create_split_index():
    split = [[], [], []]
    path0 = Path('../data') #datasets/Twibot-20
    split_list = pd.read_csv(path0 / 'split.csv')
    label = pd.read_csv(path0 / 'label.csv')

    users_index_to_uid = list(label['id'])
    uid_to_users_index = {x : i for i, x in enumerate(users_index_to_uid)}
    for id in split_list[split_list['split'] == 'train']['id']:
        split[0].append(uid_to_users_index[id])
    for id in split_list[split_list['split'] == 'val']['id']:
        split[1].append(uid_to_users_index[id])
    for id in split_list[split_list['split'] == 'test']['id']:
        split[2].append(uid_to_users_index[id])

    return split


def stack():
    split = create_split_index()
    train_idx = split[0] + split[1]
    test_idx = split[2]

    data = pd.read_csv("../data/features.csv")
    y_oof = np.load("../data/y_oof.npy")
    y_test = np.load("../data/y_test.npy")
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    train['roberta_pred'] = y_oof.tolist()
    test['roberta_pred'] = y_test.tolist()
    train.to_csv("../data/stack_train.csv", index=False)
    test.to_csv("../data/stack_test.csv", index=False)



if __name__ == '__main__':
    generate_tweet_embedding('roberta-large')
