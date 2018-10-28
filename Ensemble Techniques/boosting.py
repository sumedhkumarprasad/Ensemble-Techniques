import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

#Set the working directory
os.chdir("E:\K2_Analytics\Works\R_backup\ppt_codes")
os.listdir()



BSDF_dev = pd.read_csv("DEV_SAMPLE.csv")
X_cont =  BSDF_dev[['Age', 'Gender', 'Balance', 'Occupation',
               'No_OF_CR_TXNS', 'AGE_BKT', 'SCR', 'Holding_Period']]
y = BSDF_dev["Target"]

BSDF_holdout = pd.read_csv("E:/K2_Analytics/Works/R_backup/ppt_codes/HOLDOUT_SAMPLE.csv")
X_holdout =  BSDF_holdout[['Age', 'Gender', 'Balance', 'Occupation',
               'No_OF_CR_TXNS', 'AGE_BKT', 'SCR', 'Holding_Period']]

y_test = BSDF_holdout["Target"]

#Categorical Variable to Numerical Variables
X = pd.get_dummies(X_cont)
X_test = pd.get_dummies(X_holdout)
X.columns

model = AdaBoostClassifier(n_estimators=30)
model.fit(X, y)

pred_y_train = model.predict(X)
pred_y_train

## Let us see the classification accuracy of our model
score = accuracy_score(y, pred_y_train)
score


pred_y_test = model.predict(X_test)
pred_y_test

## Let us see the classification accuracy of our model
score_test = accuracy_score(y_test, pred_y_test)
score_test


y_train_prob = model.predict_proba(X)
fpr, tpr, thresholds =  roc_curve(y, y_train_prob[:,1])
auc(fpr, tpr)



y_test_prob = model.predict_proba(X_test)
fpr, tpr, thresholds =  roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)


scores = model_selection.cross_val_score(model, X, y, cv = 10, scoring='roc_auc')
scores.mean()
scores.std()

param_dist = {"n_estimators":np.arange(10,20),
               "learning_rate": [0.3,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3],
              }
              
tree = AdaBoostClassifier(random_state=None)
tree_cv  = GridSearchCV(tree, param_dist,
                        scoring = 'accuracy', verbose = 100)
tree_cv.fit(X, y)


## Building the model using best combination of parameters
print("Tuned Decision Tree parameter : {}".format(tree_cv.best_params_))
classifier = tree_cv.best_estimator_
classifier.fit(X,y)


y_train_prob = classifier.predict_proba(X)
fpr, tpr, thresholds = roc_curve(y, y_train_prob[:,1])
auc_d = auc(fpr, tpr)
auc_d
y_test_prob = classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc_h = auc(fpr, tpr)
auc_h

Prediction = classifier.predict_proba(X)
BSDF_dev["prob_score"] = Prediction[:,1]

#scoring step
#decile code
def deciles(x):
    decile = pd.Series(index=[0,1,2,3,4,5,6,7,8,9])
    for i in np.arange(0.1,1.1,0.1):
        decile[int(i*10)]=x.quantile(i)
    def z(x):
        if x<decile[1]: return(1)
        elif x<decile[2]: return(2)
        elif x<decile[3]: return(3)
        elif x<decile[4]: return(4)
        elif x<decile[5]: return(5)
        elif x<decile[6]: return(6)
        elif x<decile[7]: return(7)
        elif x<decile[8]: return(8)
        elif x<decile[9]: return(9)
        elif x<=decile[10]: return(10)
        else:return(np.NaN)
    s=x.map(z)
    return(s) 


def Rank_Ordering(X,y,Target):
    X['decile']=deciles(X[y])
    Rank=X.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[Target]),
        np.size(x[Target][x[Target]==0]),
        ],
        index=(["min_resp","max_resp","avg_resp",
                "cnt","cnt_resp","cnt_non_resp"])
        )).reset_index()
    Rank = Rank.sort_values(by='decile',ascending=False)
    Rank["rrate"] = Rank["cnt_resp"]*100/Rank["cnt"]
    Rank["cum_resp"] = np.cumsum(Rank["cnt_resp"])
    Rank["cum_non_resp"] = np.cumsum(Rank["cnt_non_resp"])
    Rank["cum_resp_pct"] = Rank["cum_resp"]/np.sum(Rank["cnt_resp"])
    Rank["cum_non_resp_pct"]=Rank["cum_non_resp"]/np.sum(Rank["cnt_non_resp"])
    Rank["KS"] = Rank["cum_resp_pct"] - Rank["cum_non_resp_pct"]
    Rank
    return(Rank)

Rank = Rank_Ordering(BSDF_dev,"prob_score","Target")
Rank

## Let us see the Rank Ordering on Hold-Out Dataset
Prediction_h = classifier.predict_proba(X_test)
BSDF_holdout["prob_score"] = Prediction_h[:,1]

Rank_h = Rank_Ordering(BSDF_holdout,"prob_score","Target")
Rank_h


