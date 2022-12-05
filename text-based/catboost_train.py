
from sklearn.model_selection import KFold 
import numpy as np
from catboost import CatBoostClassifier
import pandas as pd
from utils import eval


SEED = 323

def catboost_cv_train(X_train, Y_train, X_test, seed, cat_par=None):
    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    y_oof = np.zeros(X_train.shape[0])
    y_test = np.zeros(X_test.shape[0])
    ix = 0
    for train_ind, val_ind in kf.split(X_train, Y_train):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            X_train.iloc[train_ind].reset_index(drop=True),
            X_train.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            Y_train.iloc[train_ind].reset_index(drop=True),
            Y_train.iloc[val_ind].reset_index(drop=True),
        )

        if cat_par:
            clf = CatBoostClassifier(**cat_par, thread_count=-1)
        else:
            clf = CatBoostClassifier(thread_count=-1)

        clf.fit(tr_x, tr_y, eval_set=[(val_x, val_y)], verbose=100)
        preds = clf.predict_proba(val_x)[:, 1]
        y_oof[val_ind] = y_oof[val_ind] + preds

        preds_test = clf.predict_proba(X_test)[:, 1]
        y_test = y_test + preds_test / N_FOLDS
        ix = ix + 1
        

    return y_test, y_oof

def main():
    X_train = pd.read_csv("../data/stack_train.csv")
    Y_train = X_train.label
    
    X_test = pd.read_csv("../data/stack_test.csv")
    Y_test = X_test.label

    del X_train['label']
    del X_test['label']

    cat_parms = {  
            'learning_rate':0.00462499, 
            'iterations': 2325,
            'od_wait': 200,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_strength': 0.0001,
            'bagging_temperature': 0.0001,
            'random_state':SEED
        }

    y_test, y_oof = catboost_cv_train(X_train, Y_train, X_test, SEED, cat_parms)

    y_test_cls = np.where(y_test > 0.5, 1, 0)
    y_oof_cls = np.where(y_oof > 0.5, 1, 0)

    print("Cross Validation score")
    eval(y_oof, y_oof_cls, Y_train)

    print("Test score")
    eval(y_test, y_test_cls, Y_test)

if __name__ == "__main__":
    main()

    
