from catboost import CatBoostClassifier
from sklearn.model_selection import KFold 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real,  Integer
import pickle
import time


SEED=323
N_FOLDS = 5


def main():
    X_train = pd.read_csv("../data/stack_train.csv")
    Y_train = X_train.label
    
    X_test = pd.read_csv("../data/stack_test.csv")
    Y_test = X_test.label

    del X_train['label']
    del X_test['label']


    clf = CatBoostClassifier(random_state=SEED, thread_count=-1)
    cat_parms = {  
                'learning_rate': Real(1e-4, 1e-1, prior='log-uniform'),
                'iterations': Integer(300,3000),
                'depth': Integer(4,8),
                'l2_leaf_reg': Real(1, 3, prior='log-uniform'),
                'random_strength': Real(1e-4, 1, prior='log-uniform'),
                'bagging_temperature': Real(1e-4, 1, prior='log-uniform'),
            }

    fit_params = {
        'verbose': False
    }   

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    number_of_iteration = 300
    
    opt = BayesSearchCV(clf, cat_parms, n_iter=number_of_iteration, fit_params=fit_params, n_jobs=-1, random_state=SEED, cv=kf)
    
    
    # Start the search!
    start_time = time.time()
    _ = opt.fit(X_train, Y_train)
    duration = time.time() - start_time
    
    save_model_path = "catboost_bsearch.pkl"
    with open('log.txt', 'a') as f:
        f.write(f"Search for {number_of_iteration} iteration. Time used: {duration:.4f}\n")
        f.write(f"Best CV score: {opt.best_score_}\n")
        f.write(f"Best parameter: {opt.best_params_}\n")
        f.write(f"Save model in {save_model_path }\n\n")
    pickle.dump(opt.best_estimator_, open('/data/hchangac/ML/TwiModel/save_model/'+ save_model_path , 'wb'))
    



if __name__ == "__main__":
    main()
    
